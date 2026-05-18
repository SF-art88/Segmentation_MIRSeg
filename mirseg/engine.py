from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .constants import IGNORE_INDEX, SETTING_PRE
from .data import GliomaMRIDataset, collate_batch
from .losses import MIRSegReconciliationLoss, SettingAwareSegmentationLoss
from .metrics import dice_score
from .model import MIRSeg, SharedBaseline3D
from .transforms import apply_missing_mask, apply_random_corruption, sample_subset_mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def build_model(cfg: Mapping[str, Any]) -> MIRSeg:
    model_cfg = dict(cfg.get("model", {}))
    name = str(model_cfg.get("name", "mirseg")).lower()
    backbone = dict(model_cfg.get("backbone", {}))
    heads = dict(model_cfg.get("heads", {}))
    kwargs = {
        "stem_channels": int(backbone.get("stem_channels", 32)),
        "encoder_channels": tuple(int(v) for v in backbone.get("encoder_channels", [64, 96, 128, 160])),
        "decoder_channels": int(backbone.get("decoder_channels", 96)),
        "num_classes_pre": int(heads.get("num_classes_pre", 4)),
        "num_classes_post": int(heads.get("num_classes_post", 5)),
        "dropout": float(backbone.get("dropout", 0.0)),
    }
    if name in {"baseline", "shared_baseline"}:
        return SharedBaseline3D(**kwargs)
    if name in {"mirseg", "unicir", "unicir_3d"}:
        return MIRSeg(
            **kwargs,
            deep_supervision=str(model_cfg.get("deep_supervision", "learned")),
            enable_subset_correction=bool(model_cfg.get("enable_subset_correction", True)),
        )
    raise ValueError(f"Unsupported model.name={name!r}; public release includes baseline and MIRSeg only")


def build_dataset(cfg: Mapping[str, Any], split: str, training: bool) -> GliomaMRIDataset:
    data = dict(cfg.get("data", {}))
    manifest = data.get("manifest")
    if not manifest:
        raise ValueError("data.manifest must point to a public manifest CSV")
    label_maps = data.get("label_maps", {})
    return GliomaMRIDataset(
        manifest,
        split=split,
        patch_size=data.get("patch_size", [96, 96, 96]) if training else None,
        training=training,
        positive_crop_probability=float(data.get("positive_crop_probability", 0.7)),
        percentile_clip=tuple(data.get("percentile_clip", [0.5, 99.5])) if data.get("percentile_clip") else None,
        normalize_nonzero=bool(data.get("normalize_nonzero", True)),
        ignore_index=int(data.get("ignore_index", IGNORE_INDEX)),
        label_maps=label_maps,
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def make_reduced_batch(batch: Mapping[str, torch.Tensor], cfg: Mapping[str, Any], epoch: int, rng: random.Random) -> dict[str, torch.Tensor]:
    train_cfg = dict(cfg.get("train", {}))
    cur = dict(train_cfg.get("curriculum", {}))
    missing_cfg = dict(cur.get("missing", {}))
    corruption_cfg = dict(cur.get("corruption", {}))
    tier_c_start = int(cur.get("tier_c_start_epoch", 16))
    tiers: list[str] = ["B"] if epoch < tier_c_start else list(missing_cfg.get("tiers", ["B", "C"]))
    tier_probs = missing_cfg.get("tier_probabilities", None)

    image = batch["image"].detach().clone()
    present = batch["modality_present"].detach().clone().to(torch.bool)
    b = image.shape[0]
    subset_images = []
    subset_present = []
    for i in range(b):
        mask, _, _ = sample_subset_mask(present[i].cpu(), tiers=tiers, tier_probabilities=tier_probs, rng=rng)
        x_i, a_i = apply_missing_mask(image[i].cpu(), present[i].cpu(), mask.cpu())
        subset_images.append(x_i)
        subset_present.append(a_i)
    subset_image = torch.stack(subset_images, dim=0).to(image.device)
    subset_present_tensor = torch.stack(subset_present, dim=0).to(present.device)

    corruption_start = int(corruption_cfg.get("start_epoch", 10**9))
    moderate_start = int(corruption_cfg.get("moderate_start_epoch", 10**9))
    if epoch >= corruption_start:
        severity = tuple(corruption_cfg.get("moderate_severity_range", [0.35, 0.70])) if epoch >= moderate_start else tuple(corruption_cfg.get("mild_severity_range", [0.10, 0.35]))
        max_modalities = int(corruption_cfg.get("moderate_max_modalities", 2)) if epoch >= moderate_start else int(corruption_cfg.get("mild_max_modalities", 1))
        p_case = float(corruption_cfg.get("p_case", 0.5))
        types = list(corruption_cfg.get("types", ["motion", "bias", "noise", "contrast", "registration", "blur", "gibbs"]))
        corrupted_images = []
        corrupted = []
        corruption_type = []
        corruption_severity = []
        for i in range(b):
            xi, ci, ti, si = apply_random_corruption(subset_image[i].cpu(), subset_present_tensor[i].cpu(), types, severity, max_modalities, p_case, rng)
            corrupted_images.append(xi)
            corrupted.append(ci)
            corruption_type.append(ti)
            corruption_severity.append(si)
        subset_image = torch.stack(corrupted_images, dim=0).to(image.device)
        batch = dict(batch)
        batch["modality_corrupted"] = torch.stack(corrupted, dim=0).to(image.device)
        batch["corruption_type"] = torch.stack(corruption_type, dim=0).to(image.device)
        batch["corruption_severity"] = torch.stack(corruption_severity, dim=0).to(image.device)

    reduced = dict(batch)
    reduced["subset_image"] = subset_image
    reduced["subset_present"] = subset_present_tensor
    return reduced


def active_logits(outputs: Mapping[str, Any], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    setting = batch["setting_id"].long().view(-1)
    pre = outputs["pre_logits"]
    post = outputs["post_logits"]
    max_classes = max(pre.shape[1], post.shape[1])
    out = pre.new_zeros((pre.shape[0], max_classes, *pre.shape[-3:]))
    pre_idx = torch.nonzero(setting == 0, as_tuple=False).flatten()
    post_idx = torch.nonzero(setting == 1, as_tuple=False).flatten()
    if pre_idx.numel():
        out[pre_idx, : pre.shape[1]] = pre.index_select(0, pre_idx)
    if post_idx.numel():
        out[post_idx, : post.shape[1]] = post.index_select(0, post_idx)
    return out


@torch.no_grad()
def validate(model: MIRSeg, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["image"], batch["modality_present"])
        logits = active_logits(outputs, batch)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        target = batch["target"].cpu().numpy()
        setting = batch["setting_id"].cpu().numpy()
        for i in range(pred.shape[0]):
            labels = [1, 2, 3] if int(setting[i]) == SETTING_PRE else [1, 2, 3, 4]
            rows.append(dice_score(pred[i], target[i], labels))
    return {"mean_dice": float(np.nanmean(rows)) if rows else float("nan")}


def save_checkpoint(path: Path, model: MIRSeg, optimizer: torch.optim.Optimizer, epoch: int, metrics: Mapping[str, float], cfg: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": dict(metrics), "config": dict(cfg)}, path)


def load_model_checkpoint(model: MIRSeg, checkpoint_path: str | Path, strict: bool = False) -> dict[str, Any]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected), "checkpoint": ckpt}


def train_from_config(config_path: str | Path) -> None:
    cfg = load_yaml(config_path)
    runtime = dict(cfg.get("runtime", {}))
    train_cfg = dict(cfg.get("train", {}))
    set_seed(int(runtime.get("seed", 42)))
    if runtime.get("num_threads"):
        torch.set_num_threads(int(runtime["num_threads"]))
    device = torch.device(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = build_dataset(cfg, split=str(cfg.get("data", {}).get("train_split", "train")), training=True)
    val_ds = build_dataset(cfg, split=str(cfg.get("data", {}).get("val_split", "val")), training=False)
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg.get("batch_size", 2)), shuffle=True, num_workers=int(runtime.get("num_workers", 0)), collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(runtime.get("num_workers", 0)), collate_fn=collate_batch)

    model = build_model(cfg).to(device)
    if train_cfg.get("warmstart_checkpoint"):
        info = load_model_checkpoint(model, train_cfg["warmstart_checkpoint"], strict=False)
        print(f"Loaded warm-start checkpoint with {len(info['missing_keys'])} missing and {len(info['unexpected_keys'])} unexpected keys")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)), eps=float(train_cfg.get("eps", 1e-8)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(train_cfg.get("num_epochs", 60))), eta_min=float(train_cfg.get("min_lr", 1e-6)))
    seg_loss = SettingAwareSegmentationLoss(ignore_index=int(cfg.get("data", {}).get("ignore_index", IGNORE_INDEX)))
    rec_loss = MIRSegReconciliationLoss(
        segmentation=seg_loss,
        feature_weight=float(cfg.get("loss", {}).get("feature_weight", 1.0)),
        logit_weight=float(cfg.get("loss", {}).get("logit_weight", 1.0)),
        logit_temperature=float(cfg.get("loss", {}).get("logit_temperature", 1.0)),
    )
    output_dir = Path(train_cfg.get("output_dir", "experiments/mirseg"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(cfg.get("model", {}).get("name", "mirseg")).lower()
    full_only_epochs = int(train_cfg.get("curriculum", {}).get("full_only_epochs", 8))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 5.0))
    rng = random.Random(int(runtime.get("seed", 42)))
    best = -float("inf")
    history = []

    for epoch in range(int(train_cfg.get("num_epochs", 60))):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False)
        running = []
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if model_name in {"baseline", "shared_baseline"} or epoch < full_only_epochs:
                outputs = model(batch["image"], batch["modality_present"], apply_subset_correction=False)
                loss, logs = seg_loss(outputs, batch)
            else:
                reduced = make_reduced_batch(batch, cfg, epoch=epoch, rng=rng)
                paired = model.forward_teacher_student(batch["image"], batch["modality_present"], reduced["subset_image"], reduced["subset_present"])
                loss, logs = rec_loss(paired, batch)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{np.mean(running):.4f}")
        scheduler.step()
        val_metrics = validate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(running)), **val_metrics}
        history.append(row)
        print(json.dumps(row))
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch + 1, val_metrics, cfg)
        if val_metrics["mean_dice"] > best:
            best = val_metrics["mean_dice"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch + 1, val_metrics, cfg)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
