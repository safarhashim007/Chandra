"""Spatial coverage of correspondence points."""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPoint, box


def _points(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    return points[np.isfinite(points).all(axis=1)]


def grid_coverage(
    points_xy: np.ndarray, image_size: tuple[int, int], grid_size: tuple[int, int] = (8, 8)
) -> float:
    """Return occupied-cell fraction; image_size is (height, width), points are (x, y)."""
    height, width = image_size
    rows, cols = grid_size
    if height <= 0 or width <= 0 or rows <= 0 or cols <= 0:
        raise ValueError("image and grid dimensions must be positive")
    points = _points(points_xy)
    points = points[
        (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    ]
    if len(points) == 0:
        return 0.0
    occupied = set(
        zip(
            np.minimum((points[:, 1] * rows / height).astype(int), rows - 1),
            np.minimum((points[:, 0] * cols / width).astype(int), cols - 1),
            strict=True,
        )
    )
    return len(occupied) / (rows * cols)


def hull_coverage(points_xy: np.ndarray, image_size: tuple[int, int]) -> float:
    """Return inlier convex-hull area divided by image area."""
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    points = _points(points_xy)
    points = points[
        (points[:, 0] >= 0)
        & (points[:, 0] <= width)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height)
    ]
    if len(points) < 3:
        return 0.0
    area = MultiPoint(points.tolist()).convex_hull.intersection(box(0, 0, width, height)).area
    return max(0.0, min(1.0, area / (width * height)))


def spatial_coverage(
    points_xy: np.ndarray, image_size: tuple[int, int], grid_size: tuple[int, int] = (8, 8)
) -> dict[str, float]:
    return {
        "grid_coverage": grid_coverage(points_xy, image_size, grid_size),
        "hull_coverage": hull_coverage(points_xy, image_size),
    }
