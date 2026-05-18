from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .constants import IGNORE_INDEX, SETTING_POST, SETTING_PRE


def _resize_target(target: torch.Tensor, spatial_shape: Sequence[int], ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    if tuple(target.shape[-3:]) == tuple(spatial_shape):
        return target.long()
    y = target.float().unsqueeze(1)
    y = F.interpolate(y, size=tuple(spatial_shape), mode="nearest")[:, 0]
    return y.long()


def dice_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    include_background: bool = False,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_classes = int(logits.shape[1])
    target = target.long()
    valid = target != int(ignore_index)
    if not bool(valid.any().item()):
        return logits.sum() * 0.0

    ce = F.cross_entropy(logits, target.clamp_min(0), ignore_index=ignore_index)
    probs = torch.softmax(logits, dim=1)
    target_safe = torch.where(valid, target, torch.zeros_like(target))
    one_hot = F.one_hot(target_safe.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 4, 1, 2, 3).to(probs.dtype)
    mask = valid.unsqueeze(1).to(probs.dtype)
    probs = probs * mask
    one_hot = one_hot * mask
    class_range = range(num_classes) if include_background else range(1, num_classes)
    dice_terms: list[torch.Tensor] = []
    for c in class_range:
        p = probs[:, c]
        t = one_hot[:, c]
        denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
        score = (2.0 * (p * t).sum(dim=(1, 2, 3)) + eps) / (denom + eps)
        dice_terms.append(1.0 - score.mean())
    dice = torch.stack(dice_terms).mean() if dice_terms else ce * 0.0
    return float(dice_weight) * dice + float(ce_weight) * ce


class SettingAwareSegmentationLoss(nn.Module):
    """Setting-routed Dice+CE loss over pre/post head pyramids."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        ignore_index: int = IGNORE_INDEX,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.ignore_index = int(ignore_index)
        self.include_background = bool(include_background)

    def _head_loss(self, pyramid: Sequence[torch.Tensor], target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for k, logits in enumerate(pyramid):
            y = _resize_target(target, logits.shape[-3:], self.ignore_index)
            losses.append(weights[k] * dice_ce_loss(
                logits,
                y,
                ignore_index=self.ignore_index,
                include_background=self.include_background,
                dice_weight=self.dice_weight,
                ce_weight=self.ce_weight,
            ))
        return torch.stack(losses).sum()

    def forward(self, outputs: Mapping[str, Any], batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        target = batch["target"].long()
        setting = batch["setting_id"].long().view(-1)
        weights = outputs.get("deep_supervision_weights")
        if weights is None:
            weights = torch.ones(len(outputs["pre_pyramid"]), device=target.device) / len(outputs["pre_pyramid"])
        weights = weights.to(device=target.device, dtype=torch.float32)

        losses: list[torch.Tensor] = []
        logs: dict[str, float] = {}
        pre_idx = torch.nonzero(setting == SETTING_PRE, as_tuple=False).flatten()
        post_idx = torch.nonzero(setting == SETTING_POST, as_tuple=False).flatten()
        if pre_idx.numel():
            loss_pre = self._head_loss([x.index_select(0, pre_idx) for x in outputs["pre_pyramid"]], target.index_select(0, pre_idx), weights)
            losses.append(loss_pre)
            logs["loss_pre"] = float(loss_pre.detach().cpu())
        if post_idx.numel():
            loss_post = self._head_loss([x.index_select(0, post_idx) for x in outputs["post_pyramid"]], target.index_select(0, post_idx), weights)
            losses.append(loss_post)
            logs["loss_post"] = float(loss_post.detach().cpu())
        if not losses:
            zero = target.float().sum() * 0.0
            return zero, {"loss_supervised": 0.0}
        total = torch.stack(losses).mean()
        logs["loss_supervised"] = float(total.detach().cpu())
        return total, logs


def _cosine_feature_loss(student: torch.Tensor, teacher: torch.Tensor, chunk_tokens: int = 32768) -> torch.Tensor:
    s = student.flatten(2).transpose(1, 2)
    t = teacher.detach().flatten(2).transpose(1, 2)
    if s.shape != t.shape:
        raise ValueError(f"Feature shapes differ: {tuple(s.shape)} vs {tuple(t.shape)}")
    losses: list[torch.Tensor] = []
    for start in range(0, s.shape[1], int(chunk_tokens)):
        s_chunk = F.normalize(s[:, start : start + chunk_tokens], dim=-1)
        t_chunk = F.normalize(t[:, start : start + chunk_tokens], dim=-1)
        losses.append(1.0 - (s_chunk * t_chunk).sum(dim=-1).mean())
    return torch.stack(losses).mean()


def _sym_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0, ignore_mask: torch.Tensor | None = None) -> torch.Tensor:
    tau = float(temperature)
    s_log = F.log_softmax(student_logits / tau, dim=1)
    t_log = F.log_softmax(teacher_logits.detach() / tau, dim=1)
    s_prob = s_log.exp()
    t_prob = t_log.exp()
    kl_st = (s_prob * (s_log - t_log)).sum(dim=1)
    kl_ts = (t_prob * (t_log - s_log)).sum(dim=1)
    loss = 0.5 * (kl_st + kl_ts) * (tau ** 2)
    if ignore_mask is not None:
        valid = ignore_mask.to(loss.device, dtype=torch.bool)
        if tuple(valid.shape[-3:]) != tuple(loss.shape[-3:]):
            valid = _resize_target(valid.long(), loss.shape[-3:], ignore_index=0).bool()
        valid_f = valid.to(loss.dtype)
        return (loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    return loss.mean()


class MIRSegReconciliationLoss(nn.Module):
    """Supervised loss plus feature/logit same-case fuller-to-subset reconciliation."""

    def __init__(
        self,
        segmentation: SettingAwareSegmentationLoss | None = None,
        feature_weight: float = 1.0,
        logit_weight: float = 1.0,
        feature_paths: Sequence[str] = ("bottleneck", "decoder_pyramid:0", "decoder_pyramid:1"),
        feature_path_weights: Sequence[float] = (1.0, 1.0, 0.5),
        logit_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.segmentation = segmentation or SettingAwareSegmentationLoss()
        self.feature_weight = float(feature_weight)
        self.logit_weight = float(logit_weight)
        self.feature_paths = tuple(feature_paths)
        self.feature_path_weights = tuple(float(v) for v in feature_path_weights)
        self.logit_temperature = float(logit_temperature)

    @staticmethod
    def _resolve_path(outputs: Mapping[str, Any], path: str) -> torch.Tensor:
        if ":" in path:
            key, idx = path.split(":", 1)
            return outputs[key][int(idx)]
        return outputs[path]

    def _feature_loss(self, student: Mapping[str, Any], teacher: Mapping[str, Any]) -> torch.Tensor:
        losses = []
        for path, weight in zip(self.feature_paths, self.feature_path_weights):
            losses.append(float(weight) * _cosine_feature_loss(self._resolve_path(student, path), self._resolve_path(teacher, path)))
        return torch.stack(losses).sum() / max(1e-6, sum(self.feature_path_weights))

    def _logit_loss(self, student: Mapping[str, Any], teacher: Mapping[str, Any], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        setting = batch["setting_id"].long().view(-1)
        target = batch["target"].long()
        losses: list[torch.Tensor] = []
        for head_name, setting_id in (("pre_pyramid", SETTING_PRE), ("post_pyramid", SETTING_POST)):
            idx = torch.nonzero(setting == setting_id, as_tuple=False).flatten()
            if not idx.numel():
                continue
            for s_logits, t_logits in zip(student[head_name], teacher[head_name]):
                y = _resize_target(target.index_select(0, idx), s_logits.shape[-3:], IGNORE_INDEX)
                valid = y != IGNORE_INDEX
                losses.append(_sym_kl(s_logits.index_select(0, idx), t_logits.index_select(0, idx), self.logit_temperature, ignore_mask=valid))
        if not losses:
            return target.float().sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, paired_outputs: Mapping[str, Any], batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        teacher = paired_outputs["teacher"]
        student = paired_outputs["student"]
        supervised, logs = self.segmentation(student, batch)
        feature = self._feature_loss(student, teacher) if self.feature_weight else supervised * 0.0
        logit = self._logit_loss(student, teacher, batch) if self.logit_weight else supervised * 0.0
        total = supervised + self.feature_weight * feature + self.logit_weight * logit
        logs.update({
            "loss_feature": float(feature.detach().cpu()),
            "loss_logit": float(logit.detach().cpu()),
            "loss_total": float(total.detach().cpu()),
        })
        return total, logs
