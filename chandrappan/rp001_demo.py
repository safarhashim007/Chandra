"""Shared, deterministic utilities for the experimental RP-001 demo only.

Nothing in this module changes the official RoMa v2 inference path.  It holds
the image preparation and manifest helpers used by the regional-specialist
builder, trainer, and static demo renderer so their inputs remain identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

PREPROCESSING_METHODS = ("raw", "robust", "clahe", "gamma")


def sha256_file(path: Path) -> str:
    """Return the SHA256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    """Hash a JSON-compatible object with a stable representation."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_values(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("science crop contains no finite intensity values")
    return finite


def _percentile_scale(
    image: np.ndarray, low_percentile: float, high_percentile: float
) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = _finite_values(values)
    low, high = np.percentile(finite, (low_percentile, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("science crop has a degenerate intensity range")
    return np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0.0, 1.0)


def preprocess_science(image: np.ndarray, method: str) -> np.ndarray:
    """Create a deterministic [0, 1] RoMa input from a science-raster crop.

    ``raw`` deliberately reproduces the existing RP-001 RoMa input convention
    (per-crop 1st/99th percentile scaling).  The other modes are controlled
    alternatives evaluated only on TRAIN/VALIDATION before a run is selected.
    """
    if method not in PREPROCESSING_METHODS:
        raise ValueError(f"unsupported RP-001 preprocessing method: {method}")
    if method == "raw":
        return _percentile_scale(image, 1.0, 99.0)
    if method == "robust":
        return _percentile_scale(image, 2.0, 98.0)
    normalized = _percentile_scale(image, 1.0, 99.0)
    if method == "clahe":
        uint8 = np.rint(normalized * 255.0).astype(np.uint8)
        return (
            cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(uint8).astype(np.float32)
            / 255.0
        )
    # A single fixed gamma is deterministic and avoids validation-set tuning of
    # a continuous photometric parameter.
    return np.power(normalized, 0.85, dtype=np.float32)


def photometric_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply mild intensity-only augmentation; it never changes geometry."""
    brightness = float(rng.uniform(-0.05, 0.05))
    contrast = float(rng.uniform(0.92, 1.08))
    gamma = float(rng.uniform(0.92, 1.08))
    augmented = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    return np.power(augmented, gamma, dtype=np.float32)


def as_roma_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert one normalized science crop to a BCHW, three-channel tensor."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("RoMa image must be a finite two-dimensional normalized crop")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("RoMa image must be normalized to [0, 1]")
    return torch.from_numpy(np.repeat(values[None], 3, axis=0))[None].to(device)
