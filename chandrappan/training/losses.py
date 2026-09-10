"""Mask-safe building blocks for LunarRoMa supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not bool(mask.any()):
        return values.new_zeros(())
    return values[mask].mean()


def warp_huber_loss(
    predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """Smooth-L1 correspondence loss; invalid and negative-pair warps are excluded."""
    if predicted.shape != target.shape or predicted.shape[-1] != 2:
        raise ValueError("predicted and target warps must share an x/y final dimension")
    error = F.smooth_l1_loss(predicted, target, reduction="none", beta=beta).mean(dim=-1)
    return _masked_mean(error, valid)


def roma_robust_warp_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    stride: int,
    alpha: float = 0.5,
    c: float = 1e-3,
) -> torch.Tensor:
    """RoMa robust Euclidean warp loss for one refiner stride.

    This is the public RoMa regression form: with ``epe`` in normalized RoMa
    coordinates and ``cs = c * stride``, it is
    ``cs**alpha * ((epe / cs)**2 + 1)**(alpha / 2)``.  The released RoMa v2
    package does not ship a public trainer; this preserves the source RoMa
    objective used by its predecessor instead of silently retaining the
    project-only Smooth-L1 surrogate.
    """
    if predicted.shape != target.shape or predicted.shape[-1] != 2:
        raise ValueError("predicted and target warps must share an x/y final dimension")
    if stride not in {1, 2, 4}:
        raise ValueError("RoMa v2 refiner stride must be one of 1, 2, or 4")
    if not 0.0 < alpha <= 2.0 or c <= 0.0:
        raise ValueError("alpha must be in (0, 2] and c must be positive")
    cs = c * stride
    epe = torch.linalg.vector_norm(predicted - target, dim=-1)
    loss = cs**alpha * ((epe / cs).square() + 1.0).pow(alpha / 2.0)
    return _masked_mean(loss, valid)


def overlap_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("overlap logits and target must have the same shape")
    return F.binary_cross_entropy_with_logits(logits, target.float())


def coarse_nll_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Cross entropy for valid overlapping source tokens only."""
    if logits.ndim < 2:
        raise ValueError("coarse logits require a class dimension")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_valid = valid.reshape(-1).bool()
    if not bool(flat_valid.any()):
        return logits.new_zeros(())
    return F.cross_entropy(flat_logits[flat_valid], flat_targets[flat_valid])


def total_refiner_loss(
    predicted_warp: torch.Tensor,
    target_warp: torch.Tensor,
    valid: torch.Tensor,
    overlap_logits: torch.Tensor,
    overlap_target: torch.Tensor,
    precision_loss: torch.Tensor | None = None,
    *,
    stride: int = 1,
    use_legacy_huber: bool = False,
    huber_beta: float = 1.0,
    overlap_weight: float = 1e-2,
    precision_weight: float = 1e-3,
) -> dict[str, torch.Tensor]:
    warp = (
        warp_huber_loss(predicted_warp, target_warp, valid, beta=huber_beta)
        if use_legacy_huber
        else roma_robust_warp_loss(predicted_warp, target_warp, valid, stride=stride)
    )
    overlap = overlap_bce_loss(overlap_logits, overlap_target)
    precision = precision_loss if precision_loss is not None else warp.new_zeros(())
    total = warp + overlap_weight * overlap + precision_weight * precision
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite LunarRoMa refiner loss")
    return {"total": total, "warp": warp, "overlap": overlap, "precision": precision}
