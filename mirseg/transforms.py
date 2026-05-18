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
