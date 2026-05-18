# MIRSeg: Missing-Input Reconciliation for 3D Glioma MRI Segmentation

This repository contains a cleaned public implementation of **MIRSeg** (also developed under the UniCIR-3D naming in the internal codebase): a setting-aware 3D glioma MRI segmentation model for reduced-observability deployment. The release keeps the core method implementation and removes internal cluster scripts, private paths, generated outputs, run logs, ablation wrappers, and unrelated prototype branches.

The code is intended for research use with preprocessed, co-registered 3D multiparametric MRI volumes. It does not include data, trained weights, private manifests, or institution-specific preprocessing pipelines.

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

This release was distilled from an audited research environment. The public code requires only PyTorch, basic numerical/scientific Python packages, NIfTI I/O, YAML config parsing, and progress reporting. The full internal environment was not copied because it included private cluster paths, editable internal packages, and many packages unrelated to the public MIRSeg implementation.

### Recommended conda installation

The audited environment used Python 3.10.13, PyTorch 2.5.1, and `pytorch-cuda=12.4`. The public `environment.yml` keeps the core runtime/training packages and removes private prefixes and unrelated dependencies.

```bash
conda env create -f environment.yml
conda activate mirseg
```

Sanity-check the installation:

```bash
python - <<'PY'
import torch
import nibabel
import numpy
import scipy
import yaml
import tqdm

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

### Optional pip installation

Use a clean Python 3.10 environment. The audited pip environment recorded `torch==2.5.1`, but pip requirement files do not encode the CUDA wheel selector. For GPU systems, install a PyTorch 2.5.1 wheel compatible with your local CUDA/driver stack, then install the remaining packages.

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

| Package | Version recorded in audit | Public-release use |
|---|---:|---|
| Python | 3.10.13 | Runtime used for the audit; public env pins Python 3.10. |
| PyTorch | 2.5.1 | Required for model training and inference. |
| `pytorch-cuda` | 12.4 | Included in the recommended conda GPU environment because it was recorded in the audit. |
| NumPy | 1.26.4 | Required for array handling and metric aggregation. |
| SciPy | 1.15.3 | Required for connected components and surface-distance metrics. |
| NiBabel | 5.3.2 | Required for NIfTI image loading and saving. |
| PyYAML | 6.0.2 | Required for configuration files. |
| tqdm | 4.67.1 | Required for progress bars during training/evaluation. |

The audit also recorded MONAI 1.5.0, torchvision 0.20.1, torchaudio 2.5.1, scikit-learn 1.6.1, pandas 2.2.3, matplotlib 3.9.1, SimpleITK 2.2.1, einops 0.8.1, and other packages. These are not included in the public minimal dependency files because they are not required by the current public MIRSeg release.

### CUDA and PyTorch compatibility notes

The audited conda environment contained `pytorch=2.5.1` with `pytorch-cuda=12.4`. It also contained several lower-level CUDA runtime packages from the internal solver. The public environment keeps only the PyTorch CUDA metapackage and does not specify GPU model, driver version, or cluster module requirements, because those details were not provided as portable public requirements.

For CPU-only use or for systems using a different CUDA stack, adjust only the PyTorch installation while keeping the remaining package versions unchanged.

### Reproducibility notes

The public installation files are intentionally not a full clone of the internal environment. They remove private paths, editable internal packages, GADI-specific prefixes, and unrelated analysis/debugging packages. Numerical reproduction of paper results still depends on using the same data preprocessing, manifest/split definitions, model configuration, random seeds, and checkpoints. The environment files only define the software dependencies needed to run the released code.

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

## Citation note

The associated manuscript is a NeurIPS submission titled:

```text
MIRSeg: Missing-Input Reconciliation for 3D Glioma MRI Segmentation under Reduced Observability
```

Please cite the public paper version when it becomes available. A formal BibTeX entry is not included because the archival citation has not been provided in the release materials.

## Limitations

This repository is a cleaned public research-code release, not a full clinical deployment package. It intentionally omits private manifests, private data paths, HPC/PBS scripts, generated experiment outputs, internal logs, and unrelated prototype branches. The code assumes preprocessed, co-registered NIfTI inputs and does not reproduce hidden preprocessing pipelines. No trained weights are included. Stress testing through synthetic masking/corruption should not be interpreted as proof of robustness to prospectively acquired incomplete clinical protocols. A license file is not included because no release license was provided; add one before public distribution.
