# Public MIRSeg repository file contents

This document contains the complete content of every recommended public-facing file.

## `.gitignore`

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.env
.venv/
venv/
checkpoints/
experiments/
outputs/
predictions/
logs/
*.pt
*.pth
*.ckpt
*.nii
*.nii.gz
*.npz
*.npy
.DS_Store
```

## `README.md`

# MIRSeg: Missing-Input Reconciliation for 3D Glioma MRI Segmentation

This repository contains a public implementation of **MIRSeg**: a setting-aware 3D glioma MRI segmentation model for reduced-observability deployment. The release keeps the core method implementation. The code is intended for research use with preprocessed, co-registered 3D multiparametric MRI volumes. It does not include data, trained weights, private manifests, or institution-specific preprocessing pipelines.

## Method summary

MIRSeg uses a fixed four-sequence input interface in the order:

```text
T1, T1ce, T2, FLAIR
```

The model explicitly receives a modality-availability vector. Missing channels may be zero-filled as placeholders, but missingness is never inferred from image intensity alone. Degraded-but-present channels remain marked as available and are tracked separately during corrupted-present stress construction.

The architecture contains:

- modality-specific shallow 3D stems,
- availability-aware fusion,
- a dual-resolution 3D encoder for local lesion detail and broader anatomical context,
- a pyramid decoder with deep supervision,
- separate pre-treatment and post-treatment heads,
- a lightweight subset-correction adapter that is active only when modalities are missing.

During MIRSeg training, a fuller same-case forward is used as a stop-gradient teacher and a reduced-input forward is trained as the deployable student. The student is optimized with setting-routed Dice+CE supervision plus feature-level and logit-level reconciliation. At inference time, only one deployment path is used; there is no modality synthesis, subset-specific model bank, or test-time adaptation.

## Repository structure

```text
configs/
  baseline.yaml             # shared setting-aware baseline
  mirseg.yaml               # MIRSeg reduced-observability training
examples/
  manifest_template.csv     # safe manifest template with placeholder paths
mirseg/
  constants.py              # modality order, class names, tiers, corruption IDs
  data.py                   # manifest-driven NIfTI dataset
  transforms.py             # missing-modality and corrupted-present transforms
  model.py                  # baseline and MIRSeg architecture
  losses.py                 # setting-aware segmentation and reconciliation losses
  metrics.py                # voxel, lesion-wise, boundary metrics
  inference.py              # sliding-window inference
  engine.py                 # training loop and checkpoint handling
scripts/
  train.py                  # train baseline or MIRSeg
  infer.py                  # run inference from a manifest
  evaluate.py               # evaluate predicted masks
requirements.txt
README.md
```

## Installation

### Recommended conda installation

The environment used Python 3.10.13, PyTorch 2.5.1, and `pytorch-cuda=12.4`.

```bash
conda env create -f environment.yml
conda activate mirseg
```

### Optional pip installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch must be installed separately for your platform, install it first, then install the non-PyTorch dependencies:

```bash
pip install numpy==1.26.4 scipy==1.15.3 nibabel==5.3.2 PyYAML==6.0.2 tqdm==4.67.1
```

### Key package versions

| Package | Version | Public-release use |
|---|---:|---|
| Python | 3.10.13 | Runtime used for the audit; public env pins Python 3.10. |
| PyTorch | 2.5.1 | Required for model training and inference. |
| NumPy | 1.26.4 | Required for array handling and metric aggregation. |
| SciPy | 1.15.3 | Required for connected components and surface-distance metrics. |
| NiBabel | 5.3.2 | Required for NIfTI image loading and saving. |
| PyYAML | 6.0.2 | Required for configuration files. |

Other packages includes MONAI 1.5.0, torchvision 0.20.1, torchaudio 2.5.1, scikit-learn 1.6.1, pandas 2.2.3, matplotlib 3.9.1, SimpleITK 2.2.1, and einops 0.8.1.

### Reproducibility notes

## Expected data format

The public code expects preprocessed NIfTI files that are already aligned across modalities for each case. The release does not perform skull stripping, atlas registration, or institution-specific preprocessing.

Each manifest row represents one study/timepoint. Required columns are:

| column | description |
|---|---|
| `case_id` | unique case identifier used for output filenames |
| `subject_id` | subject identifier for reporting; may equal `case_id` |
| `split` | `train`, `val`, `test`, or another split name used by commands |
| `setting_id` | `0` for pre-treatment, `1` for post-treatment |
| `path_t1` | path to T1 NIfTI, or empty if unavailable |
| `path_t1ce` | path to T1ce NIfTI, or empty if unavailable |
| `path_t2` | path to T2 NIfTI, or empty if unavailable |
| `path_flair` | path to FLAIR NIfTI, or empty if unavailable |
| `path_target_pre` | pre-treatment segmentation label path; empty for post-treatment cases |
| `path_target_post` | post-treatment segmentation label path; empty for pre-treatment cases |

Default label conventions are configurable in the YAML files. The shipped defaults are:

- pre-treatment: `0=background`, `1=edema`, `2=non-enhancing/necrotic core`, `3=enhancing tumor`, after remapping BraTS 2021 raw labels `{0:0, 1:2, 2:1, 4:3}`;
- post-treatment: `0=background`, `1=enhancing tissue`, `2=non-enhancing tumor core`, `3=surrounding non-enhancing FLAIR hyperintensity`, `4=resection cavity`, assuming contiguous labels.

## Training

First train the shared setting-aware baseline:

```bash
python scripts/train.py --config configs/baseline.yaml
```

Then train MIRSeg, optionally warm-starting from the baseline checkpoint:

```bash
# edit configs/mirseg.yaml so data.manifest points to your CSV
# optionally set train.warmstart_checkpoint: experiments/baseline/best.pt
python scripts/train.py --config configs/mirseg.yaml
```

Checkpoints are written to `train.output_dir` as `last.pt` and `best.pt`.

## Inference

Run sliding-window inference on a manifest split:

```bash
python scripts/infer.py \
  --config configs/mirseg.yaml \
  --checkpoint experiments/mirseg/best.pt \
  --manifest /path/to/manifest.csv \
  --split test \
  --out-dir predictions/mirseg_test
```

Predictions are saved as one NIfTI file per case using the pattern:

```text
predictions/mirseg_test/<case_id>.nii.gz
```

## Evaluation

Evaluate predicted masks against the target paths in the manifest:

```bash
python scripts/evaluate.py \
  --manifest /path/to/manifest.csv \
  --pred-dir predictions/mirseg_test \
  --split test \
  --out outputs/mirseg_test_metrics.json
```

The evaluator reports mean Dice, lesion-wise precision/sensitivity/F1, false-positive components, HD95, and Surface Dice. Metrics are computed separately only when the corresponding split contains pre- or post-treatment cases.

## Missing-modality and corrupted-present stress

The public implementation includes the stress families used by the method:

- Tier A: all four sequences present;
- Tier B: one sequence missing;
- Tier C: two visible sequences using `T1ce+FLAIR`, `T2+FLAIR`, `T1+T1ce`, and `T1+FLAIR`;
- corrupted-present families: motion, bias field, noise, contrast attenuation, registration offset, blur/downsampling, and Gibbs/ghosting.

During MIRSeg training, the curriculum begins with full-input supervised training, then activates missing-modality reconciliation, and later introduces corrupted-present augmentation. The default curriculum is defined in `configs/mirseg.yaml`.

## License

This project is released under the Apache License 2.0.

This repository is intended for research and reproducibility purposes. It has not been validated, certified, or approved as a clinical medical device.

## Citation note

The associated manuscript is a NeurIPS submission titled:

```text
MIRSeg: Missing-Input Reconciliation for 3D Glioma MRI Segmentation under Reduced Observability
```

Please cite the public paper version when it becomes available. 

## `configs/baseline.yaml`

```yaml
# Shared setting-aware baseline used to initialize MIRSeg.
model:
  name: baseline
  backbone:
    stem_channels: 32
    encoder_channels: [64, 96, 128, 160]
    decoder_channels: 96
    dropout: 0.0
  heads:
    num_classes_pre: 4
    num_classes_post: 5

data:
  manifest: examples/manifest_template.csv  # TODO: replace with your manifest CSV
  train_split: train
  val_split: val
  patch_size: [96, 96, 96]
  positive_crop_probability: 0.7
  percentile_clip: [0.5, 99.5]
  normalize_nonzero: true
  ignore_index: -1
  label_maps:
    pre: {0: 0, 1: 2, 2: 1, 4: 3}
    post: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

runtime:
  seed: 42
  device: cuda
  num_workers: 0
  num_threads: 4
  sliding_window_roi: [96, 96, 96]
  sliding_window_overlap: 0.5

train:
  output_dir: experiments/baseline
  batch_size: 2
  num_epochs: 50
  lr: 3.0e-4
  weight_decay: 1.0e-4
  eps: 1.0e-8
  min_lr: 1.0e-6
  max_grad_norm: 5.0
  curriculum:
    full_only_epochs: 50
```

## `configs/mirseg.yaml`

```yaml
# Public MIRSeg configuration. Replace data.manifest with your local CSV.
model:
  name: mirseg
  backbone:
    stem_channels: 32
    encoder_channels: [64, 96, 128, 160]
    decoder_channels: 96
    dropout: 0.0
  heads:
    num_classes_pre: 4
    num_classes_post: 5
  deep_supervision: learned
  enable_subset_correction: true

data:
  manifest: examples/manifest_template.csv  # TODO: replace with your manifest CSV
  train_split: train
  val_split: val
  patch_size: [96, 96, 96]
  positive_crop_probability: 0.7
  percentile_clip: [0.5, 99.5]
  normalize_nonzero: true
  ignore_index: -1
  label_maps:
    # BraTS 2021 raw labels: 0 background, 1 NCR/NET, 2 edema, 4 enhancing tumor.
    pre: {0: 0, 1: 2, 2: 1, 4: 3}
    # BraTS-PTG/post-treatment labels are expected to be contiguous here.
    post: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

loss:
  feature_weight: 1.0
  logit_weight: 1.0
  logit_temperature: 1.0

runtime:
  seed: 42
  device: cuda
  num_workers: 0
  num_threads: 4
  sliding_window_roi: [96, 96, 96]
  sliding_window_overlap: 0.5

train:
  output_dir: experiments/mirseg
  batch_size: 2
  num_epochs: 60
  lr: 3.0e-4
  weight_decay: 1.0e-4
  eps: 1.0e-8
  min_lr: 1.0e-6
  max_grad_norm: 5.0
  warmstart_checkpoint: ""  # Optional: path to a baseline best.pt.
  curriculum:
    full_only_epochs: 8
    tier_c_start_epoch: 16
    missing:
      tiers: [B, C]
      tier_probabilities: {B: 0.60, C: 0.40}
    corruption:
      start_epoch: 36
      moderate_start_epoch: 46
      p_case: 0.5
      mild_max_modalities: 1
      moderate_max_modalities: 2
      mild_severity_range: [0.10, 0.35]
      moderate_severity_range: [0.35, 0.70]
      types: [motion, bias, noise, contrast, registration, blur, gibbs]
```

## `environment.yml`

```yaml
name: mirseg
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch>=2.1
  - numpy>=1.24
  - scipy>=1.10
  - nibabel>=5.0
  - pyyaml>=6.0
  - tqdm>=4.66
```

## `examples/manifest_template.csv`

```csv
case_id,subject_id,split,setting_id,path_t1,path_t1ce,path_t2,path_flair,path_target_pre,path_target_post
TODO_CASE_PRE_001,TODO_SUBJECT_001,train,0,/path/to/TODO_CASE_PRE_001_t1.nii.gz,/path/to/TODO_CASE_PRE_001_t1ce.nii.gz,/path/to/TODO_CASE_PRE_001_t2.nii.gz,/path/to/TODO_CASE_PRE_001_flair.nii.gz,/path/to/TODO_CASE_PRE_001_seg.nii.gz,
TODO_CASE_POST_001,TODO_SUBJECT_002,val,1,/path/to/TODO_CASE_POST_001_t1.nii.gz,/path/to/TODO_CASE_POST_001_t1ce.nii.gz,/path/to/TODO_CASE_POST_001_t2.nii.gz,/path/to/TODO_CASE_POST_001_flair.nii.gz,,/path/to/TODO_CASE_POST_001_seg.nii.gz
```

## `mirseg/__init__.py`

```python
"""Public MIRSeg implementation."""

from .constants import MODALITY_ORDER, PRE_CLASS_NAMES, POST_CLASS_NAMES
from .model import MIRSeg, SharedBaseline3D

__all__ = ["MIRSeg", "SharedBaseline3D", "MODALITY_ORDER", "PRE_CLASS_NAMES", "POST_CLASS_NAMES"]
```

## `mirseg/constants.py`

```python
from __future__ import annotations

from enum import IntEnum

MODALITY_ORDER: tuple[str, str, str, str] = ("T1", "T1ce", "T2", "FLAIR")
NUM_MODALITIES: int = 4

SETTING_PRE: int = 0
SETTING_POST: int = 1
IGNORE_INDEX: int = -1

PRE_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "edema",
    "non_enhancing_or_necrotic_core",
    "enhancing_tumor",
)

POST_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "enhancing_tissue",
    "non_enhancing_tumor_core",
    "surrounding_non_enhancing_flair_hyperintensity",
    "resection_cavity",
)

# Paper stress tiers. Tier A is full input, Tier B removes one sequence,
# Tier C uses plausible two-visible subsets.
MISSINGNESS_PATTERNS: dict[str, dict[str, tuple[bool, bool, bool, bool]]] = {
    "A": {
        "T1+T1ce+T2+FLAIR": (True, True, True, True),
    },
    "B": {
        "T1ce+T2+FLAIR": (False, True, True, True),
        "T1+T2+FLAIR": (True, False, True, True),
        "T1+T1ce+FLAIR": (True, True, False, True),
        "T1+T1ce+T2": (True, True, True, False),
    },
    "C": {
        "T1ce+FLAIR": (False, True, False, True),
        "T2+FLAIR": (False, False, True, True),
        "T1+T1ce": (True, True, False, False),
        "T1+FLAIR": (True, False, False, True),
    },
}

CORRUPTION_TYPES: tuple[str, ...] = (
    "motion",
    "bias",
    "noise",
    "contrast",
    "registration",
    "blur",
    "gibbs",
)


class CorruptionType(IntEnum):
    NONE = 0
    MOTION = 1
    BIAS = 2
    NOISE = 3
    CONTRAST = 4
    REGISTRATION = 5
    BLUR = 6
    GIBBS = 7


CORRUPTION_NAME_TO_ID: dict[str, int] = {
    "none": int(CorruptionType.NONE),
    "motion": int(CorruptionType.MOTION),
    "bias": int(CorruptionType.BIAS),
    "noise": int(CorruptionType.NOISE),
    "contrast": int(CorruptionType.CONTRAST),
    "registration": int(CorruptionType.REGISTRATION),
    "blur": int(CorruptionType.BLUR),
    "gibbs": int(CorruptionType.GIBBS),
}
```

## `mirseg/data.py`

```python
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
```

## `mirseg/engine.py`

```python
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
```

## `mirseg/inference.py`

```python
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
```

## `mirseg/losses.py`

```python
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
```

## `mirseg/metrics.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .constants import IGNORE_INDEX
from .data import load_manifest


def dice_score(pred: np.ndarray, target: np.ndarray, labels: Sequence[int]) -> float:
    scores = []
    for label in labels:
        p = pred == int(label)
        t = target == int(label)
        denom = p.sum() + t.sum()
        if denom == 0:
            continue
        scores.append((2.0 * np.logical_and(p, t).sum()) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def foreground_binary(mask: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    for label in labels:
        out |= mask == int(label)
    return out


def lesionwise_detection(pred_fg: np.ndarray, target_fg: np.ndarray) -> dict[str, float]:
    pred_lab, n_pred = ndi.label(pred_fg)
    target_lab, n_target = ndi.label(target_fg)
    tp = 0
    matched_pred: set[int] = set()
    for idx in range(1, n_target + 1):
        overlap = np.unique(pred_lab[target_lab == idx])
        overlap = [int(v) for v in overlap if int(v) != 0]
        if overlap:
            tp += 1
            matched_pred.update(overlap)
    fp = max(0, int(n_pred) - len(matched_pred))
    fn = max(0, int(n_target) - tp)
    precision = tp / max(1, tp + fp)
    sensitivity = tp / max(1, tp + fn)
    f1 = 2 * precision * sensitivity / max(1e-8, precision + sensitivity)
    return {"lesion_precision": precision, "lesion_sensitivity": sensitivity, "lesion_f1": f1, "fp_components": float(fp)}


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def surface_distances(a: np.ndarray, b: np.ndarray, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    if not a.any() and not b.any():
        return np.asarray([0.0], dtype=np.float32)
    if not a.any() or not b.any():
        return np.asarray([np.inf], dtype=np.float32)
    sa, sb = _surface(a), _surface(b)
    dt_b = ndi.distance_transform_edt(~sb, sampling=spacing)
    dt_a = ndi.distance_transform_edt(~sa, sampling=spacing)
    return np.concatenate([dt_b[sa], dt_a[sb]]).astype(np.float32)


def hd95(pred_fg: np.ndarray, target_fg: np.ndarray, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    d = surface_distances(pred_fg, target_fg, spacing)
    return float(np.percentile(d, 95)) if np.isfinite(d).any() else float("inf")


def surface_dice(pred_fg: np.ndarray, target_fg: np.ndarray, tolerance: float = 1.0, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    if not pred_fg.any() and not target_fg.any():
        return 1.0
    if not pred_fg.any() or not target_fg.any():
        return 0.0
    sp = _surface(pred_fg)
    st = _surface(target_fg)
    dt_t = ndi.distance_transform_edt(~st, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    ok_p = (dt_t[sp] <= tolerance).sum()
    ok_t = (dt_p[st] <= tolerance).sum()
    denom = sp.sum() + st.sum()
    return float((ok_p + ok_t) / max(1, denom))


def evaluate_pair(pred: np.ndarray, target: np.ndarray, labels: Sequence[int], spacing: Sequence[float] = (1.0, 1.0, 1.0), surface_tolerance: float = 1.0) -> dict[str, float]:
    valid = target != IGNORE_INDEX
    if not np.any(valid):
        return {}
    pred_eval = np.where(valid, pred, 0)
    target_eval = np.where(valid, target, 0)
    pred_fg = foreground_binary(pred_eval, labels) & valid
    target_fg = foreground_binary(target_eval, labels) & valid
    out = {"dice": dice_score(pred_eval[valid], target_eval[valid], labels)}
    out.update(lesionwise_detection(pred_fg, target_fg))
    out["hd95"] = hd95(pred_fg, target_fg, spacing)
    out["surface_dice"] = surface_dice(pred_fg, target_fg, surface_tolerance, spacing)
    return out


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for row in rows for k in row})
    return {k: float(np.nanmean([row[k] for row in rows if k in row])) for k in keys}


def evaluate_prediction_directory(
    manifest_csv: str | Path,
    pred_dir: str | Path,
    split: str,
    labels_pre: Sequence[int] = (1, 2, 3),
    labels_post: Sequence[int] = (1, 2, 3, 4),
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    pred_dir = Path(pred_dir)
    rows = []
    by_setting = {"pre": [], "post": []}
    for record in load_manifest(manifest_csv, split=split):
        target_path = record.target_path
        if target_path is None:
            continue
        pred_path = pred_dir / f"{record.case_id}.nii.gz"
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        pred_img = nib.load(str(pred_path))
        target_img = nib.load(str(target_path))
        pred = np.asarray(pred_img.get_fdata(), dtype=np.int64)
        target = np.asarray(target_img.get_fdata(), dtype=np.int64)
        labels = labels_pre if record.setting_id == 0 else labels_post
        metrics = evaluate_pair(pred, target, labels, spacing=target_img.header.get_zooms()[:3])
        metrics["case_id"] = record.case_id  # type: ignore[assignment]
        rows.append(metrics)
        by_setting["pre" if record.setting_id == 0 else "post"].append({k: v for k, v in metrics.items() if isinstance(v, float)})
    result = {
        "overall": aggregate_metrics([{k: v for k, v in r.items() if isinstance(v, float)} for r in rows]),
        "pre": aggregate_metrics(by_setting["pre"]) if by_setting["pre"] else {},
        "post": aggregate_metrics(by_setting["post"]) if by_setting["post"] else {},
        "cases": rows,
    }
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(result, indent=2))
    return result
```

## `mirseg/model.py`

```python
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
```

## `mirseg/transforms.py`

```python
from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .constants import CORRUPTION_NAME_TO_ID, MISSINGNESS_PATTERNS, NUM_MODALITIES


def available_patterns(tiers: Sequence[str]) -> list[tuple[str, str, tuple[bool, bool, bool, bool]]]:
    out: list[tuple[str, str, tuple[bool, bool, bool, bool]]] = []
    for tier in tiers:
        if tier not in MISSINGNESS_PATTERNS:
            raise KeyError(f"Unknown missingness tier {tier!r}. Expected one of {sorted(MISSINGNESS_PATTERNS)}")
        for name, mask in MISSINGNESS_PATTERNS[tier].items():
            out.append((tier, name, mask))
    return out


def sample_subset_mask(
    declared_present: torch.Tensor,
    tiers: Sequence[str] = ("B", "C"),
    tier_probabilities: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, str, str]:
    """Sample a realistic reduced-observability mask and intersect it with declared availability."""
    rng = rng or random
    present = declared_present.to(dtype=torch.bool)
    if present.numel() != NUM_MODALITIES:
        raise ValueError(f"declared_present must have {NUM_MODALITIES} entries, got {present.numel()}")

    tiers = tuple(str(t) for t in tiers)
    if tier_probabilities:
        weights = [float(tier_probabilities.get(t, 0.0)) for t in tiers]
        if sum(weights) <= 0:
            weights = [1.0] * len(tiers)
        tier = rng.choices(list(tiers), weights=weights, k=1)[0]
    else:
        tier = rng.choice(list(tiers))

    patterns = list(MISSINGNESS_PATTERNS[tier].items())
    name, pattern = rng.choice(patterns)
    requested = torch.tensor(pattern, dtype=torch.bool, device=present.device)
    reduced = present & requested

    # Keep at least one actually available channel. If the sampled subset conflicts
    # with declared availability, fall back to one declared-present channel.
    if not bool(reduced.any().item()):
        available = torch.nonzero(present, as_tuple=False).flatten().tolist()
        if not available:
            reduced[:] = True
        else:
            reduced[:] = False
            reduced[int(rng.choice(available))] = True
            name = "single_declared_available_fallback"
    return reduced, str(tier), str(name)


def apply_missing_mask(
    image: torch.Tensor,
    modality_present: torch.Tensor,
    requested_present: torch.Tensor,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-fill masked channels and update the explicit availability vector."""
    if image.ndim != 4:
        raise ValueError(f"Expected image [4,D,H,W], got {tuple(image.shape)}")
    present = modality_present.to(dtype=torch.bool) & requested_present.to(dtype=torch.bool)
    out = image.clone()
    for m in range(NUM_MODALITIES):
        if not bool(present[m].item()):
            out[m].fill_(float(fill_value))
    return out, present


def _ensure_5d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x[None, None]
    if x.ndim == 4:
        return x[None]
    if x.ndim == 5:
        return x
    raise ValueError(f"Expected 3D, 4D, or 5D tensor, got {tuple(x.shape)}")


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(round(3.0 * max(float(sigma), 1e-3))))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma + 1e-6))
    return kernel / kernel.sum().clamp_min(1e-6)


def gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    x5 = _ensure_5d(x)
    device, dtype = x5.device, x5.dtype
    kernel = _gaussian_kernel1d(sigma, device=device, dtype=dtype)
    pad = kernel.numel() // 2
    groups = x5.shape[1]
    kz = kernel.view(1, 1, -1, 1, 1).repeat(groups, 1, 1, 1, 1)
    ky = kernel.view(1, 1, 1, -1, 1).repeat(groups, 1, 1, 1, 1)
    kx = kernel.view(1, 1, 1, 1, -1).repeat(groups, 1, 1, 1, 1)
    y = F.conv3d(F.pad(x5, (0, 0, 0, 0, pad, pad), mode="replicate"), kz, groups=groups)
    y = F.conv3d(F.pad(y, (0, 0, pad, pad, 0, 0), mode="replicate"), ky, groups=groups)
    y = F.conv3d(F.pad(y, (pad, pad, 0, 0, 0, 0), mode="replicate"), kx, groups=groups)
    if x.ndim == 3:
        return y[0, 0]
    if x.ndim == 4:
        return y[0]
    return y


def _motion(x: torch.Tensor, severity: float, rng: random.Random) -> torch.Tensor:
    axis = rng.choice([0, 1, 2])
    max_shift = max(1, int(round(1 + 5 * severity)))
    views = [x]
    for shift in range(1, max_shift + 1):
        views.append(torch.roll(x, shifts=shift, dims=axis))
        views.append(torch.roll(x, shifts=-shift, dims=axis))
    return torch.stack(views).mean(dim=0)


def _bias(x: torch.Tensor, severity: float) -> torch.Tensor:
    d, h, w = x.shape
    z = torch.linspace(-1.0, 1.0, d, device=x.device, dtype=x.dtype)
    y = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
    xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
    zz, yy, xxg = torch.meshgrid(z, y, xx, indexing="ij")
    coeffs = torch.randn(6, device=x.device, dtype=x.dtype)
    field = (
        1.0
        + 0.20 * severity * coeffs[0] * xxg
        + 0.20 * severity * coeffs[1] * yy
        + 0.20 * severity * coeffs[2] * zz
        + 0.15 * severity * coeffs[3] * xxg * yy
        + 0.15 * severity * coeffs[4] * xxg * zz
        + 0.15 * severity * coeffs[5] * yy * zz
    )
    field = gaussian_blur_3d(field, sigma=max(0.5, 4.0 * severity)).clamp(min=0.3)
    return x * field


def _noise(x: torch.Tensor, severity: float) -> torch.Tensor:
    nz = x[x != 0]
    scale = nz.std(unbiased=False) if nz.numel() else x.std(unbiased=False)
    if not torch.isfinite(scale) or float(scale.item()) == 0.0:
        scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return x + torch.randn_like(x) * (0.05 + 0.20 * severity) * scale


def _contrast(x: torch.Tensor, severity: float) -> torch.Tensor:
    nz = x[x != 0]
    center = nz.mean() if nz.numel() else x.mean()
    factor = max(0.25, 1.0 - 0.6 * severity)
    return center + factor * (x - center)


def _registration(x: torch.Tensor, severity: float, rng: random.Random) -> torch.Tensor:
    max_shift = max(1, int(round(1 + 4 * severity)))
    shifts = tuple(rng.randint(-max_shift, max_shift) for _ in range(3))
    return torch.roll(x, shifts=shifts, dims=(0, 1, 2))


def _blur(x: torch.Tensor, severity: float) -> torch.Tensor:
    scale = max(1.25, 1.0 + 1.5 * severity)
    d, h, w = x.shape
    low_size = (max(1, int(d / scale)), max(1, int(h / scale)), max(1, int(w / scale)))
    x5 = x[None, None]
    low = F.interpolate(x5, size=low_size, mode="trilinear", align_corners=False)
    return F.interpolate(low, size=(d, h, w), mode="trilinear", align_corners=False)[0, 0]


def _gibbs(x: torch.Tensor, severity: float, rng: random.Random) -> torch.Tensor:
    axis = rng.choice([0, 1, 2])
    shift = max(1, int(round(2 + 8 * severity)))
    ghost = torch.roll(x, shifts=shift, dims=axis)
    ringing = x + 0.15 * severity * (x - gaussian_blur_3d(x, sigma=1.0 + severity))
    return 0.85 * ringing + 0.15 * ghost


def corrupt_channel(x: torch.Tensor, corruption_type: str, severity: float, rng: random.Random | None = None) -> torch.Tensor:
    rng = rng or random
    c = corruption_type.lower()
    if c == "motion":
        return _motion(x, severity, rng)
    if c == "bias":
        return _bias(x, severity)
    if c == "noise":
        return _noise(x, severity)
    if c == "contrast":
        return _contrast(x, severity)
    if c == "registration":
        return _registration(x, severity, rng)
    if c == "blur":
        return _blur(x, severity)
    if c == "gibbs":
        return _gibbs(x, severity, rng)
    raise KeyError(f"Unsupported corruption type {corruption_type!r}")


def apply_random_corruption(
    image: torch.Tensor,
    modality_present: torch.Tensor,
    corruption_types: Sequence[str],
    severity_range: tuple[float, float],
    max_modalities: int,
    p_case: float,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply corrupted-present stress without changing modality availability."""
    rng = rng or random
    corrupted = torch.zeros(NUM_MODALITIES, dtype=torch.bool)
    corruption_id = torch.zeros(NUM_MODALITIES, dtype=torch.long)
    corruption_severity = torch.zeros(NUM_MODALITIES, dtype=torch.float32)
    if rng.random() > p_case:
        return image, corrupted, corruption_id, corruption_severity

    present_indices = [i for i, v in enumerate(modality_present.to(dtype=torch.bool).tolist()) if v]
    if not present_indices:
        return image, corrupted, corruption_id, corruption_severity
    k = min(max(1, int(max_modalities)), len(present_indices))
    chosen = rng.sample(present_indices, k=k)
    out = image.clone()
    lo, hi = float(severity_range[0]), float(severity_range[1])
    for m in chosen:
        ctype = rng.choice(list(corruption_types))
        sev = rng.uniform(lo, hi)
        out[m] = corrupt_channel(out[m], ctype, sev, rng)
        corrupted[m] = True
        corruption_id[m] = int(CORRUPTION_NAME_TO_ID[ctype])
        corruption_severity[m] = float(sev)
    return out, corrupted, corruption_id, corruption_severity
```

## `requirements.txt`

```text
torch>=2.1
numpy>=1.24
scipy>=1.10
nibabel>=5.0
PyYAML>=6.0
tqdm>=4.66
```

## `scripts/evaluate.py`

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mirseg.metrics import evaluate_prediction_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted NIfTI masks against a manifest CSV.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="outputs/eval_metrics.json")
    args = parser.parse_args()
    result = evaluate_prediction_directory(args.manifest, args.pred_dir, args.split, output_json=args.out)
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
```

## `scripts/infer.py`

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mirseg.engine import build_model, load_model_checkpoint
from mirseg.inference import run_manifest_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIRSeg sliding-window inference from a manifest CSV.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg.setdefault("data", {})["manifest"] = args.manifest
    device = torch.device(cfg.get("runtime", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg)
    load_model_checkpoint(model, args.checkpoint, strict=False)
    run_manifest_inference(
        model=model,
        manifest_csv=args.manifest,
        split=args.split,
        out_dir=args.out_dir,
        roi_size=cfg.get("runtime", {}).get("sliding_window_roi", [96, 96, 96]),
        overlap=float(cfg.get("runtime", {}).get("sliding_window_overlap", 0.5)),
        device=device,
    )


if __name__ == "__main__":
    main()
```

## `scripts/train.py`

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mirseg.engine import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the public MIRSeg or shared-baseline model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config, e.g. configs/mirseg.yaml")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
```

