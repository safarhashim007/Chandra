"""The only pixel/RoMa coordinate conversions allowed in training code."""

from __future__ import annotations

import torch


def _as_size(size: tuple[int, int]) -> tuple[float, float]:
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size {size}")
    return float(height), float(width)


def pixel_to_roma_coords(
    points_px: torch.Tensor, size: tuple[int, int], *, align_corners: bool = False
) -> torch.Tensor:
    """Convert target pixel x/y coordinates to RoMa's normalized [-1, 1] convention."""
    height, width = _as_size(size)
    if align_corners:
        scale = points_px.new_tensor((max(width - 1, 1), max(height - 1, 1)))
        return 2.0 * points_px / scale - 1.0
    scale = points_px.new_tensor((width, height))
    return 2.0 * (points_px + 0.5) / scale - 1.0


def roma_to_pixel_coords(
    points_roma: torch.Tensor, size: tuple[int, int], *, align_corners: bool = False
) -> torch.Tensor:
    height, width = _as_size(size)
    if align_corners:
        scale = points_roma.new_tensor((max(width - 1, 1), max(height - 1, 1)))
        return (points_roma + 1.0) * scale / 2.0
    scale = points_roma.new_tensor((width, height))
    return (points_roma + 1.0) * scale / 2.0 - 0.5


def gt_pixel_warp_to_roma(warp_px: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    return pixel_to_roma_coords(warp_px, target_size, align_corners=False)


def roma_warp_to_pixel(warp_roma: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    return roma_to_pixel_coords(warp_roma, target_size, align_corners=False)


def build_gt_for_stage(
    gt_warp_px: torch.Tensor,
    valid_mask: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    prediction_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample master pixel GT at prediction cell centres, then normalize for RoMa."""
    source_h, source_w = source_size
    pred_h, pred_w = prediction_shape
    if gt_warp_px.shape[-1] != 2:
        raise ValueError("GT warp must have final x/y dimension of 2")
    ys = (
        ((torch.arange(pred_h, device=gt_warp_px.device) + 0.5) * source_h / pred_h - 0.5)
        .round()
        .long()
        .clamp(0, source_h - 1)
    )
    xs = (
        ((torch.arange(pred_w, device=gt_warp_px.device) + 0.5) * source_w / pred_w - 0.5)
        .round()
        .long()
        .clamp(0, source_w - 1)
    )
    sampled = (
        gt_warp_px[..., ys[:, None], xs[None, :], :]
        if gt_warp_px.ndim == 4
        else gt_warp_px[ys[:, None], xs[None, :], :]
    )
    mask = (
        valid_mask[..., ys[:, None], xs[None, :]]
        if valid_mask.ndim == 3
        else valid_mask[ys[:, None], xs[None, :]]
    )
    return gt_pixel_warp_to_roma(sampled, target_size), mask
