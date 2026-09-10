"""Source-faithful, stage-wise supervision for the detached RoMa refiners."""

from __future__ import annotations

import torch

from chandrappan.training.coordinates import build_gt_for_stage
from chandrappan.training.losses import roma_robust_warp_loss, warp_huber_loss


def refiner_stage_losses(
    refiners: list[dict[str, object]],
    gt_warp_px: torch.Tensor,
    valid_mask: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    *,
    legacy_huber_beta: float | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise every detached refiner output against physical master GT.

    RoMa's refiner inputs are intentionally detached between stages. A loss on
    only the final output therefore cannot train strides 4 and 2. This helper
    applies a separate target and loss to each stage while preserving those
    source detach boundaries.
    """
    if not refiners:
        raise ValueError("refiner output is required for refiner supervision")
    losses: dict[str, torch.Tensor] = {}
    for stage in refiners:
        stride = stage["stride"]
        output = stage["ab"]
        if not isinstance(stride, int) or not isinstance(output, dict):
            raise TypeError("invalid forward_train refiner output")
        prediction = output["warp"]
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("refiner warp must be a tensor")
        target, valid = build_gt_for_stage(
            gt_warp_px, valid_mask, source_size, target_size, prediction.shape[1:3]
        )
        losses[f"warp_stride_{stride}"] = (
            warp_huber_loss(prediction, target, valid, beta=legacy_huber_beta)
            if legacy_huber_beta is not None
            else roma_robust_warp_loss(prediction, target, valid, stride=stride)
        )
    losses["total"] = torch.stack(list(losses.values())).sum()
    return losses
