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
