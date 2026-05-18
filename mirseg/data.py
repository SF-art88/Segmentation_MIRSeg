from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import IGNORE_INDEX, MODALITY_ORDER, NUM_MODALITIES, SETTING_POST, SETTING_PRE


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "nan"}:
        return None
    return text


@dataclass(frozen=True)
class ManifestRecord:
    case_id: str
    split: str
    setting_id: int
    path_t1: str | None
    path_t1ce: str | None
    path_t2: str | None
    path_flair: str | None
    path_target_pre: str | None
    path_target_post: str | None
    subject_id: str | None = None
    study_id: str | None = None
    timepoint_id: str | None = None
    site_id: str | None = None

    @property
    def modality_paths(self) -> tuple[str | None, str | None, str | None, str | None]:
        return (self.path_t1, self.path_t1ce, self.path_t2, self.path_flair)

    @property
    def target_path(self) -> str | None:
        return self.path_target_pre if self.setting_id == SETTING_PRE else self.path_target_post


def load_manifest(path: str | Path, split: str | Sequence[str] | None = None) -> list[ManifestRecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    wanted = {split} if isinstance(split, str) else set(split or [])
    records: list[ManifestRecord] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"case_id", "split", "setting_id", "path_t1", "path_t1ce", "path_t2", "path_flair"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Manifest {path} is missing required columns: {sorted(missing)}")
        for row in reader:
            if wanted and row.get("split") not in wanted:
                continue
            record = ManifestRecord(
                case_id=str(row["case_id"]),
                split=str(row["split"]),
                setting_id=int(row["setting_id"]),
                path_t1=_optional_text(row.get("path_t1")),
                path_t1ce=_optional_text(row.get("path_t1ce")),
                path_t2=_optional_text(row.get("path_t2")),
                path_flair=_optional_text(row.get("path_flair")),
                path_target_pre=_optional_text(row.get("path_target_pre")),
                path_target_post=_optional_text(row.get("path_target_post")),
                subject_id=_optional_text(row.get("subject_id")),
                study_id=_optional_text(row.get("study_id")),
                timepoint_id=_optional_text(row.get("timepoint_id")),
                site_id=_optional_text(row.get("site_id")),
            )
            if record.setting_id not in {SETTING_PRE, SETTING_POST}:
                raise ValueError(f"case {record.case_id}: setting_id must be 0 or 1")
            records.append(record)
    if not records:
        raise ValueError(f"No records loaded from {path} for split={split!r}")
    return records


def _load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    array = np.asarray(img.get_fdata(dtype=np.float32))
    return array, img.affine.astype(np.float32)


def _zscore_nonzero(x: np.ndarray) -> np.ndarray:
    mask = x != 0
    if not np.any(mask):
        return x.astype(np.float32, copy=False)
    vals = x[mask]
    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-6:
        std = 1.0
    out = x.copy().astype(np.float32)
    out[mask] = (out[mask] - mean) / std
    return out


def _percentile_clip(x: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = x != 0
    if not np.any(mask):
        return x
    lo, hi = np.percentile(x[mask], [low, high])
    return np.clip(x, lo, hi)


def _remap_labels(label: np.ndarray, mapping: Mapping[int, int] | None, ignore_index: int) -> np.ndarray:
    label = label.astype(np.int64, copy=False)
    if mapping is None:
        return label
    out = np.full_like(label, fill_value=ignore_index, dtype=np.int64)
    for src, dst in mapping.items():
        out[label == int(src)] = int(dst)
    missing_values = sorted(set(np.unique(label).tolist()) - set(int(k) for k in mapping))
    missing_values = [v for v in missing_values if v != ignore_index]
    if missing_values:
        raise ValueError(f"Target contains labels not covered by remap: {missing_values}")
    return out


def _crop_slices(shape: Sequence[int], center: Sequence[int], patch_size: Sequence[int]) -> tuple[slice, slice, slice]:
    slices: list[slice] = []
    for dim, c, p in zip(shape, center, patch_size):
        p = min(int(p), int(dim))
        start = int(c) - p // 2
        start = max(0, min(start, int(dim) - p))
        slices.append(slice(start, start + p))
    return tuple(slices)  # type: ignore[return-value]


def _pad_to_patch(image: torch.Tensor, target: torch.Tensor, patch_size: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    pads: list[int] = []
    for current, wanted in zip(reversed(image.shape[-3:]), reversed(patch_size)):
        deficit = max(0, int(wanted) - int(current))
        pads.extend([deficit // 2, deficit - deficit // 2])
    if any(pads):
        image = torch.nn.functional.pad(image, pads, mode="constant", value=0.0)
        target = torch.nn.functional.pad(target[None, None].float(), pads, mode="constant", value=float(IGNORE_INDEX))[0, 0].long()
    return image, target


class GliomaMRIDataset(Dataset):
    """Manifest-driven NIfTI dataset for MIRSeg.

    The public release assumes that images are already co-registered into a common
    voxel space. It intentionally does not include institution-specific preprocessing
    paths or private split-generation logic.
    """

    def __init__(
        self,
        manifest: str | Path,
        split: str = "train",
        patch_size: Sequence[int] | None = (96, 96, 96),
        training: bool = True,
        positive_crop_probability: float = 0.7,
        percentile_clip: tuple[float, float] | None = (0.5, 99.5),
        normalize_nonzero: bool = True,
        ignore_index: int = IGNORE_INDEX,
        label_maps: Mapping[str, Mapping[int, int]] | None = None,
    ) -> None:
        self.records = load_manifest(manifest, split=split)
        self.patch_size = tuple(int(v) for v in patch_size) if patch_size else None
        self.training = bool(training)
        self.positive_crop_probability = float(positive_crop_probability)
        self.percentile_clip = percentile_clip
        self.normalize_nonzero = bool(normalize_nonzero)
        self.ignore_index = int(ignore_index)
        self.label_maps = dict(label_maps or {})

    def __len__(self) -> int:
        return len(self.records)

    def _load_image_stack(self, record: ManifestRecord) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        arrays: list[np.ndarray] = []
        present: list[bool] = []
        affine: np.ndarray | None = None
        reference_shape: tuple[int, int, int] | None = None

        for path in record.modality_paths:
            if path is None:
                arrays.append(None)  # type: ignore[arg-type]
                present.append(False)
                continue
            array, this_affine = _load_nifti(path)
            if array.ndim != 3:
                raise ValueError(f"{record.case_id}: expected 3D NIfTI, got shape {array.shape} for {path}")
            if reference_shape is None:
                reference_shape = tuple(array.shape)
                affine = this_affine
            elif tuple(array.shape) != reference_shape:
                raise ValueError(f"{record.case_id}: modality shapes do not match")
            if self.percentile_clip is not None:
                array = _percentile_clip(array, *self.percentile_clip)
            if self.normalize_nonzero:
                array = _zscore_nonzero(array)
            arrays.append(array.astype(np.float32, copy=False))
            present.append(True)

        if reference_shape is None:
            raise ValueError(f"{record.case_id}: no input modality path was provided")
        filled: list[np.ndarray] = []
        for array in arrays:
            if array is None:
                filled.append(np.zeros(reference_shape, dtype=np.float32))
            else:
                filled.append(array)
        image = torch.from_numpy(np.stack(filled, axis=0)).float()
        return image, torch.tensor(present, dtype=torch.bool), np.eye(4, dtype=np.float32) if affine is None else affine

    def _load_target(self, record: ManifestRecord, shape: Sequence[int]) -> torch.Tensor:
        target_path = record.target_path
        if target_path is None:
            return torch.full(tuple(shape), self.ignore_index, dtype=torch.long)
        target, _ = _load_nifti(target_path)
        if tuple(target.shape) != tuple(shape):
            raise ValueError(f"{record.case_id}: target shape {target.shape} does not match image shape {shape}")
        key = "pre" if record.setting_id == SETTING_PRE else "post"
        target = _remap_labels(target, self.label_maps.get(key), self.ignore_index)
        return torch.from_numpy(target).long()

    def _random_crop(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.patch_size is None:
            return image, target
        image, target = _pad_to_patch(image, target, self.patch_size)
        shape = target.shape[-3:]
        use_positive = self.training and random.random() < self.positive_crop_probability and bool((target > 0).any().item())
        if use_positive:
            coords = torch.nonzero(target > 0, as_tuple=False)
            center = coords[random.randrange(coords.shape[0])].tolist()
        else:
            center = [random.randrange(int(s)) for s in shape]
        slices = _crop_slices(shape, center, self.patch_size)
        return image[(slice(None),) + slices], target[slices]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, modality_present, affine = self._load_image_stack(record)
        target = self._load_target(record, image.shape[-3:])
        if self.training:
            image, target = self._random_crop(image, target)
        return {
            "case_id": record.case_id,
            "subject_id": record.subject_id or record.case_id,
            "image": image,
            "modality_present": modality_present,
            "modality_corrupted": torch.zeros(NUM_MODALITIES, dtype=torch.bool),
            "corruption_type": torch.zeros(NUM_MODALITIES, dtype=torch.long),
            "corruption_severity": torch.zeros(NUM_MODALITIES, dtype=torch.float32),
            "setting_id": torch.tensor(record.setting_id, dtype=torch.long),
            "target": target,
            "affine": torch.from_numpy(affine),
            "path_target": record.target_path or "",
            "split": record.split,
        }


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keys_to_stack = {
        "image",
        "modality_present",
        "modality_corrupted",
        "corruption_type",
        "corruption_severity",
        "setting_id",
        "target",
        "affine",
    }
    batch: dict[str, Any] = {}
    for key in samples[0].keys():
        values = [s[key] for s in samples]
        if key in keys_to_stack:
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values
    return batch
