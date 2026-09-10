"""Pixel-space correspondence metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def correspondence_metrics(
    predicted_px: torch.Tensor,
    target_px: torch.Tensor,
    valid: torch.Tensor,
    thresholds: Iterable[float] = (1.0, 3.0, 5.0, 10.0),
) -> dict[str, float | int]:
    if predicted_px.shape != target_px.shape or predicted_px.shape[-1] != 2:
        raise ValueError("predicted and target pixels must have identical x/y shapes")
    errors = torch.linalg.vector_norm(predicted_px - target_px, dim=-1)[valid.bool()]
    if errors.numel() == 0:
        raise ValueError("cannot compute correspondence metrics without valid points")
    result: dict[str, float | int] = {
        "valid_correspondences": int(errors.numel()),
        "median_epe_px": float(errors.median().item()),
        "mean_epe_px": float(errors.mean().item()),
    }
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("PCK thresholds must be finite and positive")
        result[f"pck_{threshold:g}"] = float((errors <= threshold).float().mean().item())
    return result
