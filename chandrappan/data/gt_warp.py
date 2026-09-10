"""Generate dense master ground truth in physical target pixel coordinates."""

from __future__ import annotations

import numpy as np

from chandrappan.geo.metadata import LunarImageMetadata


def dense_pixel_warp(
    source: LunarImageMetadata, target: LunarImageMetadata
) -> tuple[np.ndarray, np.ndarray]:
    """Map every source pixel centre to target pixels and return (warp_px, valid)."""
    y, x = np.indices((source.height, source.width), dtype=np.float64)
    world_x, world_y = source.transform.pixel_to_world(x, y)
    target_x, target_y = target.transform.world_to_pixel(world_x, world_y)
    warp = np.stack((target_x, target_y), axis=-1).astype(np.float32)
    valid = (
        (target_x >= 0) & (target_x < target.width) & (target_y >= 0) & (target_y < target.height)
    )
    return warp, valid
