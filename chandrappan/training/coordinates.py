"""The only pixel/RoMa coordinate conversions allowed in training code."""

from __future__ import annotations

import torch
from torch.nn import functional as F


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
    return 2.0 * points_px / scale - 1.0


def roma_to_pixel_coords(
    points_roma: torch.Tensor, size: tuple[int, int], *, align_corners: bool = False
) -> torch.Tensor:
    height, width = _as_size(size)
    if align_corners:
        scale = points_roma.new_tensor((max(width - 1, 1), max(height - 1, 1)))
        return (points_roma + 1.0) * scale / 2.0
    scale = points_roma.new_tensor((width, height))
    return (points_roma + 1.0) * scale / 2.0


def gt_pixel_warp_to_roma(warp_px: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    return pixel_to_roma_coords(warp_px, target_size, align_corners=False)


def roma_warp_to_pixel(warp_roma: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    return roma_to_pixel_coords(warp_roma, target_size, align_corners=False)


def normalized_delta_to_pixel(delta_roma: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Convert an x/y RoMa-coordinate displacement to target-pixel displacement."""
    height, width = _as_size(size)
    return delta_roma * delta_roma.new_tensor((width / 2.0, height / 2.0))


def refiner_head_delta_to_pixel(
    raw_delta: torch.Tensor, *, refine_init: float = 4.0
) -> torch.Tensor:
    """Convert official ConvRefiner head units to pixels.

    ConvRefiner adds ``raw / (refine_init * (W, H))`` in normalized
    coordinates. RoMa's normalized-to-pixel conversion consequently makes a
    raw head displacement worth ``1 / (2 * refine_init)`` pixels on either
    axis, independent of the refiner stride.
    """
    if refine_init <= 0:
        raise ValueError("refine_init must be positive")
    return raw_delta / (2.0 * refine_init)


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
    if gt_warp_px.ndim == 3:
        gt_warp_px = gt_warp_px.unsqueeze(0)
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0)
    if gt_warp_px.shape[:3] != valid_mask.shape:
        raise ValueError("GT warp and valid mask must share batch/height/width")
    if gt_warp_px.shape[1:3] != (source_h, source_w):
        raise ValueError("source_size must match the master GT dimensions")

    # align_corners=False maps -1+1/N and 1-1/N to pixel centres. Sampling
    # bilinearly preserves an affine physical GT rather than introducing a
    # nearest-neighbour half-pixel bias at coarse refiner stages.
    ys = torch.arange(pred_h, device=gt_warp_px.device, dtype=gt_warp_px.dtype) + 0.5
    xs = torch.arange(pred_w, device=gt_warp_px.device, dtype=gt_warp_px.dtype) + 0.5
    y_norm = 2.0 * ys / pred_h - 1.0
    x_norm = 2.0 * xs / pred_w - 1.0
    grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing="ij")
    grid = (
        torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(gt_warp_px.shape[0], -1, -1, -1)
    )
    sampled = F.grid_sample(
        gt_warp_px.permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)
    sampled_valid = F.grid_sample(
        valid_mask[:, None].float(), grid, mode="nearest", align_corners=False
    )[:, 0]
    return gt_pixel_warp_to_roma(sampled, target_size), sampled_valid.bool()
