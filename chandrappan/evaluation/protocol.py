"""Registration acceptance and VRR/FAR definitions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AcceptanceConfig:
    min_correspondences: int = 100
    min_inliers: int = 30
    min_inlier_ratio: float = 0.25
    max_reprojection_error_px: float = 3.0
    min_spatial_coverage: float = 0.10
    min_scale: float = 0.8
    max_scale: float = 1.25
    max_rotation_degrees: float = 15.0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def accept_registration(metrics: dict[str, float | int], config: AcceptanceConfig) -> bool:
    required = (
        "correspondences",
        "inliers",
        "inlier_ratio",
        "reprojection_error_px",
        "spatial_coverage",
        "scale",
        "rotation_degrees",
    )
    if any(key not in metrics for key in required):
        raise ValueError(f"registration metrics missing keys: {required}")
    return (
        metrics["correspondences"] >= config.min_correspondences
        and metrics["inliers"] >= config.min_inliers
        and metrics["inlier_ratio"] >= config.min_inlier_ratio
        and metrics["reprojection_error_px"] <= config.max_reprojection_error_px
        and metrics["spatial_coverage"] >= config.min_spatial_coverage
        and config.min_scale <= metrics["scale"] <= config.max_scale
        and abs(metrics["rotation_degrees"]) <= config.max_rotation_degrees
    )


def vrr_far(outcomes: Iterable[tuple[bool, bool]]) -> dict[str, float | int]:
    """Compute VRR on positives and FAR on negatives from (is_positive, accepted)."""
    rows = list(outcomes)
    positives = [accepted for positive, accepted in rows if positive]
    negatives = [accepted for positive, accepted in rows if not positive]
    return {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "vrr": sum(positives) / len(positives) if positives else float("nan"),
        "far": sum(negatives) / len(negatives) if negatives else float("nan"),
    }
