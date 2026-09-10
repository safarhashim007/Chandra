"""Image footprints derived directly from their pixel/world affine transforms."""

from __future__ import annotations

from .metadata import LunarImageMetadata


def footprint_corners_world(metadata: LunarImageMetadata) -> list[tuple[float, float]]:
    corners = ((0, 0), (metadata.width, 0), (metadata.width, metadata.height), (0, metadata.height))
    points = [tuple(float(v) for v in metadata.transform.pixel_to_world(x, y)) for x, y in corners]
    area_twice = sum(
        x_a * y_b - x_b * y_a for (x_a, y_a), (x_b, y_b) in zip(points, (*points[1:], points[0]))
    )
    return points if area_twice > 0 else list(reversed(points))
