from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .constants import NUM_MODALITIES, SETTING_PRE
from .data import GliomaMRIDataset, collate_batch
from .model import MIRSeg


def _compute_steps(size: int, roi: int, overlap: float) -> list[int]:
    if size <= roi:
        return [0]
    stride = max(1, int(round(roi * (1.0 - overlap))))
    steps = list(range(0, size - roi + 1, stride))
    if steps[-1] != size - roi:
        steps.append(size - roi)
    return steps


@torch.no_grad()
def sliding_window_logits(
    model: MIRSeg,
    image: torch.Tensor,
    modality_present: torch.Tensor,
    setting_id: int,
    roi_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Average-overlap sliding-window inference for one case."""
    if image.ndim != 4 or image.shape[0] != NUM_MODALITIES:
        raise ValueError(f"Expected image [4,D,H,W], got {tuple(image.shape)}")
    device = torch.device(device)
    model = model.to(device).eval()
    roi = tuple(int(v) for v in roi_size)
    d, h, w = image.shape[-3:]
    pad = [0, max(0, roi[2] - w), 0, max(0, roi[1] - h), 0, max(0, roi[0] - d)]
    x = F.pad(image.unsqueeze(0), pad).to(device)
    present = modality_present.unsqueeze(0).to(device)
    _, _, dp, hp, wp = x.shape
    z_steps = _compute_steps(dp, roi[0], overlap)
    y_steps = _compute_steps(hp, roi[1], overlap)
    x_steps = _compute_steps(wp, roi[2], overlap)
    out: torch.Tensor | None = None
    count: torch.Tensor | None = None
    head_key = "pre_logits" if int(setting_id) == SETTING_PRE else "post_logits"
    for z0 in z_steps:
        for y0 in y_steps:
            for x0 in x_steps:
                patch = x[:, :, z0 : z0 + roi[0], y0 : y0 + roi[1], x0 : x0 + roi[2]]
                logits = model(patch, present, apply_subset_correction=None)[head_key]
                if out is None:
                    out = torch.zeros((1, logits.shape[1], dp, hp, wp), device=device, dtype=logits.dtype)
                    count = torch.zeros((1, 1, dp, hp, wp), device=device, dtype=logits.dtype)
                out[:, :, z0 : z0 + roi[0], y0 : y0 + roi[1], x0 : x0 + roi[2]] += logits
                count[:, :, z0 : z0 + roi[0], y0 : y0 + roi[1], x0 : x0 + roi[2]] += 1
    assert out is not None and count is not None
    logits = out / count.clamp_min(1.0)
    return logits[:, :, :d, :h, :w][0].cpu()


def predict_mask(
    model: MIRSeg,
    image: torch.Tensor,
    modality_present: torch.Tensor,
    setting_id: int,
    roi_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    device: torch.device | str = "cuda",
) -> np.ndarray:
    logits = sliding_window_logits(model, image, modality_present, setting_id, roi_size, overlap, device)
    return torch.argmax(logits, dim=0).to(torch.int16).numpy()


def save_prediction(mask: np.ndarray, affine: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.int16), affine), str(out_path))


def run_manifest_inference(
    model: MIRSeg,
    manifest_csv: str | Path,
    split: str,
    out_dir: str | Path,
    roi_size: Sequence[int],
    overlap: float,
    device: torch.device | str,
) -> None:
    dataset = GliomaMRIDataset(manifest_csv, split=split, patch_size=None, training=False)
    out_dir = Path(out_dir)
    for sample in dataset:
        mask = predict_mask(
            model,
            sample["image"],
            sample["modality_present"],
            int(sample["setting_id"].item()),
            roi_size=roi_size,
            overlap=overlap,
            device=device,
        )
        save_prediction(mask, sample["affine"].numpy(), out_dir / f"{sample['case_id']}.nii.gz")
