from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .constants import NUM_MODALITIES


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    # Prefer several groups for normal 3D patches, but fall back to one group for
    # tiny channel counts so unit tests with very small volumes do not fail at the
    # deepest 1x1x1 level.
    if channels < max_groups:
        return nn.GroupNorm(1, channels)
    groups = max(1, min(max_groups, channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct3D(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if stride != 1 or in_channels != out_channels else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class ModalityStem3D(nn.Module):
    """Shallow slot-specific stem with learned modality scale and bias."""

    def __init__(self, out_channels: int = 32, hidden_channels: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.proj = nn.Sequential(
            ConvNormAct3D(1, hidden_channels),
            ConvNormAct3D(hidden_channels, out_channels),
        )
        self.residual = nn.Conv3d(1, out_channels, kernel_size=1, bias=False)
        self.modality_scale = nn.Embedding(NUM_MODALITIES, out_channels)
        self.modality_bias = nn.Embedding(NUM_MODALITIES, out_channels)
        nn.init.ones_(self.modality_scale.weight)
        nn.init.zeros_(self.modality_bias.weight)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"ModalityStem3D expects [B,1,D,H,W], got {tuple(x.shape)}")
        b = x.shape[0]
        ids = torch.full((b,), int(modality_id), dtype=torch.long, device=x.device)
        scale = self.modality_scale(ids).view(b, self.out_channels, 1, 1, 1)
        bias = self.modality_bias(ids).view(b, self.out_channels, 1, 1, 1)
        return self.dropout((self.proj(x) + self.residual(x)) * scale + bias)


class AvailabilityAwareFusion(nn.Module):
    """Attention fusion gated by the explicit modality availability vector."""

    def __init__(self, channels: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)
        self.modality_embedding = nn.Embedding(NUM_MODALITIES, channels)
        self.attention = nn.Sequential(
            nn.Conv3d(channels, max(8, channels), kernel_size=1),
            nn.GELU(),
            nn.Conv3d(max(8, channels), 1, kernel_size=1),
        )
        self.output_proj = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=1, bias=False), _group_norm(channels), nn.GELU())

    def forward(self, features: Sequence[torch.Tensor], modality_present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(features) != NUM_MODALITIES:
            raise ValueError(f"Expected {NUM_MODALITIES} modality features")
        stacked = torch.stack(list(features), dim=1)  # [B,M,C,D,H,W]
        b, m, c, d, h, w = stacked.shape
        present = modality_present.to(device=stacked.device, dtype=torch.bool).view(b, m)
        emb = self.modality_embedding(torch.arange(m, device=stacked.device)).view(1, m, c, 1, 1, 1)
        enriched = stacked + emb
        logits = self.attention(enriched.reshape(b * m, c, d, h, w)).mean(dim=(2, 3, 4)).view(b, m)
        logits = logits / self.temperature
        logits = logits.masked_fill(~present, float("-inf"))
        no_present = ~present.any(dim=1)
        if bool(no_present.any().item()):
            logits = logits.clone()
            logits[no_present] = 0.0
            present = present.clone()
            present[no_present] = True
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(present, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused = (stacked * weights.view(b, m, 1, 1, 1, 1)).sum(dim=1)
        return self.output_proj(fused), weights


@dataclass
class BackboneOutput:
    encoder_pyramid: list[torch.Tensor]
    bottleneck: torch.Tensor
    hr_states: list[torch.Tensor]
    lr_states: list[torch.Tensor]


class DualResolutionBackbone3D(nn.Module):
    """Four-level local/global encoder with HR-LR mutual fusion."""

    def __init__(self, in_channels: int, channels: Sequence[int] = (64, 96, 128, 160), dropout: float = 0.0) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("MIRSeg uses a four-level encoder pyramid")
        self.channels = tuple(int(v) for v in channels)
        self.hr_stem = ConvNormAct3D(in_channels, self.channels[0])
        self.lr_stem = ConvNormAct3D(in_channels, self.channels[0])
        self.hr_blocks = nn.ModuleList()
        self.lr_blocks = nn.ModuleList()
        self.hr_down = nn.ModuleList()
        self.lr_down = nn.ModuleList()
        self.lr_to_hr = nn.ModuleList()
        self.hr_to_lr = nn.ModuleList()
        self.out_proj = nn.ModuleList()
        for i, ch in enumerate(self.channels):
            in_ch = self.channels[i - 1] if i > 0 else ch
            if i > 0:
                self.hr_down.append(ConvNormAct3D(self.channels[i - 1], ch, stride=2))
                self.lr_down.append(ConvNormAct3D(self.channels[i - 1], ch, stride=2))
            self.hr_blocks.append(ResidualBlock3D(ch, ch, dropout=dropout))
            self.lr_blocks.append(ResidualBlock3D(ch, ch, dropout=dropout))
            self.lr_to_hr.append(nn.Conv3d(ch, ch, kernel_size=1, bias=False))
            self.hr_to_lr.append(nn.Conv3d(ch, ch, kernel_size=1, bias=False))
            self.out_proj.append(nn.Conv3d(ch, ch, kernel_size=1, bias=False))

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        hr = self.hr_stem(x)
        lr = self.lr_stem(F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True))
        pyramid: list[torch.Tensor] = []
        hr_states: list[torch.Tensor] = []
        lr_states: list[torch.Tensor] = []
        for i in range(4):
            if i > 0:
                hr = self.hr_down[i - 1](hr)
                lr = self.lr_down[i - 1](lr)
            hr = self.hr_blocks[i](hr)
            lr = self.lr_blocks[i](lr)
            lr_up = F.interpolate(lr, size=hr.shape[-3:], mode="trilinear", align_corners=False)
            hr_down = F.interpolate(hr, size=lr.shape[-3:], mode="trilinear", align_corners=False)
            hr = hr + self.lr_to_hr[i](lr_up)
            lr = lr + self.hr_to_lr[i](hr_down)
            pyramid.append(self.out_proj[i](hr + F.interpolate(lr, size=hr.shape[-3:], mode="trilinear", align_corners=False)))
            hr_states.append(hr)
            lr_states.append(lr)
        return BackboneOutput(encoder_pyramid=pyramid, bottleneck=pyramid[-1], hr_states=hr_states, lr_states=lr_states)


class PyramidDecoder3D(nn.Module):
    def __init__(self, encoder_channels: Sequence[int] = (64, 96, 128, 160), decoder_channels: int = 96) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv3d(c, decoder_channels, kernel_size=1, bias=False) for c in encoder_channels])
        self.refine = nn.ModuleList([ResidualBlock3D(decoder_channels, decoder_channels) for _ in encoder_channels])

    def forward(self, encoder_pyramid: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(encoder_pyramid) != 4:
            raise ValueError("Expected four encoder features")
        y = self.lateral[-1](encoder_pyramid[-1])
        outputs = [self.refine[-1](y)]
        for level in reversed(range(3)):
            y = F.interpolate(y, size=encoder_pyramid[level].shape[-3:], mode="trilinear", align_corners=False)
            y = y + self.lateral[level](encoder_pyramid[level])
            y = self.refine[level](y)
            outputs.append(y)
        return list(reversed(outputs))  # full, 1/2, 1/4, 1/8


class SettingSpecificHeads(nn.Module):
    def __init__(self, in_channels: int, num_classes_pre: int = 4, num_classes_post: int = 5) -> None:
        super().__init__()
        self.pre_heads = nn.ModuleList([nn.Conv3d(in_channels, num_classes_pre, kernel_size=1) for _ in range(4)])
        self.post_heads = nn.ModuleList([nn.Conv3d(in_channels, num_classes_post, kernel_size=1) for _ in range(4)])

    def forward(self, decoder_pyramid: Sequence[torch.Tensor]) -> dict[str, Any]:
        pre = [head(feat) for head, feat in zip(self.pre_heads, decoder_pyramid)]
        post = [head(feat) for head, feat in zip(self.post_heads, decoder_pyramid)]
        return {"pre_logits": pre[0], "post_logits": post[0], "pre_pyramid": pre, "post_pyramid": post}


class DeepSupervisionWeights(nn.Module):
    def __init__(self, mode: str = "learned", fixed_weights: Sequence[float] = (1.0, 0.5, 0.25, 0.125)) -> None:
        super().__init__()
        self.mode = str(mode)
        fixed = torch.tensor(list(fixed_weights), dtype=torch.float32)
        fixed = fixed / fixed.sum().clamp_min(1e-6)
        self.register_buffer("fixed_weights", fixed)
        self.logits = nn.Parameter(torch.zeros(len(fixed))) if self.mode == "learned" else None

    def forward(self) -> torch.Tensor:
        if self.mode == "learned":
            assert self.logits is not None
            return torch.softmax(self.logits, dim=0)
        return self.fixed_weights


class SubsetCorrectionAdapter(nn.Module):
    """Depthwise separable residual adapter active only when modalities are missing."""

    def __init__(self, channels: int, hidden_channels: int = 64, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.depthwise = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=1, bias=False),
            _group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_gate = nn.Conv3d(channels, 1, kernel_size=1)
        self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(channels, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, feature: torch.Tensor, modality_present: torch.Tensor) -> dict[str, torch.Tensor]:
        present = modality_present.to(device=feature.device, dtype=feature.dtype)
        missing_ratio = 1.0 - present.mean(dim=1, keepdim=True)
        scale = missing_ratio.view(-1, 1, 1, 1, 1) * self.residual_scale
        delta = self.pointwise(self.depthwise(feature))
        gate = torch.sigmoid(self.spatial_gate(feature)) * self.channel_gate(feature)
        corrected = feature + scale * gate * delta
        return {"feature": corrected, "delta": delta, "gate": gate, "missing_ratio": missing_ratio.squeeze(1)}


class MIRSeg(nn.Module):
    """Missing-Input Reconciliation model with a single deployment graph."""

    def __init__(
        self,
        stem_channels: int = 32,
        encoder_channels: Sequence[int] = (64, 96, 128, 160),
        decoder_channels: int = 96,
        num_classes_pre: int = 4,
        num_classes_post: int = 5,
        deep_supervision: str = "learned",
        dropout: float = 0.0,
        enable_subset_correction: bool = True,
    ) -> None:
        super().__init__()
        self.num_modalities = NUM_MODALITIES
        self.enable_subset_correction = bool(enable_subset_correction)
        self.stems = nn.ModuleList([ModalityStem3D(stem_channels, stem_channels, dropout=dropout) for _ in range(NUM_MODALITIES)])
        self.fusion = AvailabilityAwareFusion(stem_channels)
        self.backbone = DualResolutionBackbone3D(stem_channels, encoder_channels, dropout=dropout)
        self.decoder = PyramidDecoder3D(encoder_channels, decoder_channels)
        self.subset_correction = SubsetCorrectionAdapter(decoder_channels) if enable_subset_correction else None
        self.heads = SettingSpecificHeads(decoder_channels, num_classes_pre=num_classes_pre, num_classes_post=num_classes_post)
        self.deep_supervision_weights = DeepSupervisionWeights(mode=deep_supervision)

    def _apply_adapter(self, pyramid: list[torch.Tensor], modality_present: torch.Tensor, apply_subset_correction: bool | None) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        if self.subset_correction is None:
            return pyramid, {}
        should_apply = bool((modality_present.to(torch.bool).sum(dim=1) < NUM_MODALITIES).any().item()) if apply_subset_correction is None else bool(apply_subset_correction)
        if not should_apply:
            return pyramid, {}
        out = self.subset_correction(pyramid[0], modality_present)
        corrected = list(pyramid)
        corrected[0] = out["feature"]
        return corrected, out

    def forward(self, image: torch.Tensor, modality_present: torch.Tensor, apply_subset_correction: bool | None = None) -> dict[str, Any]:
        if image.ndim != 5 or image.shape[1] != NUM_MODALITIES:
            raise ValueError(f"Expected image [B,4,D,H,W], got {tuple(image.shape)}")
        stem_features = [stem(image[:, m : m + 1], modality_id=m) for m, stem in enumerate(self.stems)]
        fused, attention = self.fusion(stem_features, modality_present)
        enc = self.backbone(fused)
        raw_decoder_pyramid = self.decoder(enc.encoder_pyramid)
        decoder_pyramid, correction = self._apply_adapter(raw_decoder_pyramid, modality_present, apply_subset_correction)
        logits = self.heads(decoder_pyramid)
        out: dict[str, Any] = {
            "stem_features": stem_features,
            "fused_feature": fused,
            "modality_attention": attention,
            "encoder_pyramid": enc.encoder_pyramid,
            "bottleneck": enc.bottleneck,
            "decoder_pyramid": decoder_pyramid,
            "decoder_feature_before_subset_correction": raw_decoder_pyramid[0],
            "deep_supervision_weights": self.deep_supervision_weights(),
        }
        out.update(logits)
        if correction:
            out["subset_correction"] = correction
        return out

    def forward_teacher_student(
        self,
        full_image: torch.Tensor,
        full_present: torch.Tensor,
        subset_image: torch.Tensor,
        subset_present: torch.Tensor,
    ) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            teacher = self.forward(full_image, full_present, apply_subset_correction=False)
        self.train(was_training)
        student = self.forward(subset_image, subset_present, apply_subset_correction=True)
        return {"teacher": teacher, "student": student}


class SharedBaseline3D(MIRSeg):
    """Shared baseline used before MIRSeg reconciliation training."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("deep_supervision", "fixed")
        kwargs.setdefault("enable_subset_correction", False)
        super().__init__(*args, **kwargs)
