"""Dependency-light convex footprint intersection for catalog screening."""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]


def _inside(point: Point, edge_a: Point, edge_b: Point) -> bool:
    return (edge_b[0] - edge_a[0]) * (point[1] - edge_a[1]) - (edge_b[1] - edge_a[1]) * (
        point[0] - edge_a[0]
    ) >= 0


def _intersection(start: Point, end: Point, edge_a: Point, edge_b: Point) -> Point:
    dx1, dy1 = end[0] - start[0], end[1] - start[1]
    dx2, dy2 = edge_b[0] - edge_a[0], edge_b[1] - edge_a[1]
    denominator = dx1 * dy2 - dy1 * dx2
    if abs(denominator) < 1e-15:
        return end
    t = ((edge_a[0] - start[0]) * dy2 - (edge_a[1] - start[1]) * dx2) / denominator
    return start[0] + t * dx1, start[1] + t * dy1


def convex_intersection(subject: Sequence[Point], clip: Sequence[Point]) -> list[Point]:
    """Return a polygon intersection; inputs must be counter-clockwise convex polygons."""
    output = list(subject)
    for edge_a, edge_b in zip(clip, (*clip[1:], clip[0])):
        input_points, output = output, []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            if _inside(current, edge_a, edge_b):
                if not _inside(previous, edge_a, edge_b):
                    output.append(_intersection(previous, current, edge_a, edge_b))
                output.append(current)
            elif _inside(previous, edge_a, edge_b):
                output.append(_intersection(previous, current, edge_a, edge_b))
            previous = current
    return output
