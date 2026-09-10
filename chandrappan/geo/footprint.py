"""Image footprints derived directly from their pixel/world affine transforms."""

from __future__ import annotations

from .metadata import LunarImageMetadata


def footprint_corners_world(metadata: LunarImageMetadata) -> list[tuple[float, float]]:
    corners = ((0, 0), (metadata.width, 0), (metadata.width, metadata.height), (0, metadata.height))
    return [tuple(float(v) for v in metadata.transform.pixel_to_world(x, y)) for x, y in corners]
