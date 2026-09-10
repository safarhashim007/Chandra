"""Robust, model-aware geometric registration verification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .coverage import spatial_coverage


@dataclass(frozen=True)
class VerificationConfig:
    ransac_reprojection_threshold_px: float = 3.0
    confidence: float = 0.999
    max_iterations: int = 2000
    min_matches: int = 12
    min_inliers: int = 8
    min_inlier_ratio: float = 0.25
    max_median_reprojection_error_px: float = 3.0
    max_mean_reprojection_error_px: float = 5.0
    min_grid_coverage: float = 0.10
    min_hull_coverage: float = 0.01
    min_scale: float = 0.8
    max_scale: float = 1.25
    max_rotation_degrees: float = 15.0
    max_shear: float = 0.25
    max_anisotropy: float = 1.25
    min_determinant: float = 1e-4
    grid_size: tuple[int, int] = (8, 8)
    model_order: tuple[str, ...] = ("similarity", "affine", "homography")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TransformCandidate:
    model_type: str
    matrix: tuple[tuple[float, ...], ...] | None
    input_matches: int
    inliers: int
    inlier_ratio: float
    mean_reprojection_error_px: float
    median_reprojection_error_px: float
    max_reprojection_error_px: float
    scale: float
    rotation_degrees: float
    shear: float
    anisotropy: float
    determinant: float
    grid_coverage: float
    hull_coverage: float
    degenerate: bool
    accepted: bool
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def metrics(self) -> dict[str, float | int]:
        return {
            "correspondences": self.input_matches,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "reprojection_error_px": self.median_reprojection_error_px,
            "spatial_coverage": self.grid_coverage,
            "grid_coverage": self.grid_coverage,
            "hull_coverage": self.hull_coverage,
            "scale": self.scale,
            "rotation_degrees": self.rotation_degrees,
            "shear": self.shear,
            "anisotropy": self.anisotropy,
        }


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    selected: TransformCandidate | None
    candidates: tuple[TransformCandidate, ...]
    rejection_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "selected": self.selected.as_dict() if self.selected else None,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "rejection_reason": self.rejection_reason,
        }


def _as_points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return points


def _project(matrix: np.ndarray, source: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((source, np.ones(len(source), dtype=np.float32)))
    if matrix.shape == (2, 3):
        return homogeneous @ matrix.T
    projected = homogeneous @ matrix.T
    return projected[:, :2] / projected[:, 2:3]


def _local_linear_part(matrix: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    center = np.array([[width / 2, height / 2]], dtype=np.float32)
    basis = np.array([[width / 2 + 1, height / 2], [width / 2, height / 2 + 1]], dtype=np.float32)
    if matrix.shape == (2, 3):
        return matrix[:, :2].astype(float)
    center_target = _project(matrix, center)[0]
    basis_target = _project(matrix, basis)
    return np.column_stack((basis_target[0] - center_target, basis_target[1] - center_target))


def _transform_stats(
    matrix: np.ndarray, image_size: tuple[int, int]
) -> tuple[float, float, float, float, float, bool]:
    linear = _local_linear_part(matrix, image_size)
    if not np.isfinite(linear).all():
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), True)
    singular = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    if singular[1] <= 1e-12:
        return (float("nan"), float("nan"), float("nan"), float("nan"), determinant, True)
    scale = float(math.sqrt(abs(determinant)))
    anisotropy = float(singular[0] / singular[1])
    rotation = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
    columns = linear / np.linalg.norm(linear, axis=0)
    shear = abs(float(np.dot(columns[:, 0], columns[:, 1])))
    return scale, rotation, shear, anisotropy, determinant, False


def _fit(model: str, source: np.ndarray, target: np.ndarray, config: VerificationConfig):
    import cv2

    if model == "similarity":
        return cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_reprojection_threshold_px,
            maxIters=config.max_iterations,
            confidence=config.confidence,
        )
    if model == "affine":
        return cv2.estimateAffine2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_reprojection_threshold_px,
            maxIters=config.max_iterations,
            confidence=config.confidence,
        )
    if model == "homography":
        return cv2.findHomography(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_reprojection_threshold_px,
            maxIters=config.max_iterations,
            confidence=config.confidence,
        )
    raise ValueError(f"unsupported transform model: {model}")


def _candidate(
    model: str,
    source: np.ndarray,
    target: np.ndarray,
    image_size: tuple[int, int],
    config: VerificationConfig,
) -> TransformCandidate:
    import cv2

    required = {"similarity": 2, "affine": 3, "homography": 4}[model]
    if len(source) < max(config.min_matches, required):
        return TransformCandidate(
            model,
            None,
            len(source),
            0,
            0.0,
            math.inf,
            math.inf,
            math.inf,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            0.0,
            0.0,
            True,
            False,
            ("insufficient_matches",),
        )
    try:
        matrix, mask = _fit(model, source, target, config)
    except cv2.error:
        matrix, mask = None, None
    if matrix is None or mask is None:
        return TransformCandidate(
            model,
            None,
            len(source),
            0,
            0.0,
            math.inf,
            math.inf,
            math.inf,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            0.0,
            0.0,
            True,
            False,
            ("fit_failed",),
        )
    matrix = np.asarray(matrix, dtype=float)
    projected = _project(matrix, source)
    errors = np.linalg.norm(projected - target, axis=1)
    inlier_mask = np.asarray(mask).ravel().astype(bool)
    inlier_errors = errors[inlier_mask]
    inliers = int(inlier_mask.sum())
    coverage = spatial_coverage(source[inlier_mask], image_size, config.grid_size)
    scale, rotation, shear, anisotropy, determinant, degenerate = _transform_stats(
        matrix, image_size
    )
    reasons: list[str] = []
    if degenerate or abs(determinant) < config.min_determinant:
        reasons.append("degenerate_transform")
    if inliers < config.min_inliers:
        reasons.append("too_few_inliers")
    ratio = inliers / len(source)
    if ratio < config.min_inlier_ratio:
        reasons.append("low_inlier_ratio")
    mean_error = float(np.mean(inlier_errors)) if inliers else math.inf
    median_error = float(np.median(inlier_errors)) if inliers else math.inf
    max_error = float(np.max(inlier_errors)) if inliers else math.inf
    if median_error > config.max_median_reprojection_error_px:
        reasons.append("median_reprojection_error")
    if mean_error > config.max_mean_reprojection_error_px:
        reasons.append("mean_reprojection_error")
    if not math.isfinite(scale) or not config.min_scale <= scale <= config.max_scale:
        reasons.append("implausible_scale")
    if not math.isfinite(rotation) or abs(rotation) > config.max_rotation_degrees:
        reasons.append("implausible_rotation")
    if not math.isfinite(shear) or shear > config.max_shear:
        reasons.append("excessive_shear")
    if not math.isfinite(anisotropy) or anisotropy > config.max_anisotropy:
        reasons.append("excessive_anisotropy")
    if coverage["grid_coverage"] < config.min_grid_coverage:
        reasons.append("low_grid_coverage")
    if coverage["hull_coverage"] < config.min_hull_coverage:
        reasons.append("low_hull_coverage")
    return TransformCandidate(
        model,
        tuple(tuple(float(value) for value in row) for row in matrix),
        len(source),
        inliers,
        ratio,
        mean_error,
        median_error,
        max_error,
        scale,
        rotation,
        shear,
        anisotropy,
        determinant,
        coverage["grid_coverage"],
        coverage["hull_coverage"],
        degenerate,
        not reasons,
        tuple(reasons),
    )


def verify_registration(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    image_a_size: tuple[int, int],
    image_b_size: tuple[int, int] | None = None,
    *,
    config: VerificationConfig | None = None,
) -> VerificationResult:
    """Fit models in simplicity order and accept the first plausible model."""
    del image_b_size
    source = _as_points(source_xy, "source_xy")
    target = _as_points(target_xy, "target_xy")
    if len(source) != len(target):
        raise ValueError("source_xy and target_xy must have equal length")
    config = config or VerificationConfig()
    candidates = tuple(
        _candidate(model, source, target, image_a_size, config) for model in config.model_order
    )
    selected = next((candidate for candidate in candidates if candidate.accepted), None)
    if selected:
        return VerificationResult(True, selected, candidates, None)
    viable = [candidate for candidate in candidates if candidate.matrix is not None]
    selected = max(
        viable,
        key=lambda candidate: (candidate.inliers, -candidate.median_reprojection_error_px),
        default=None,
    )
    reason = ";".join(selected.rejection_reasons) if selected else "no_model_fit"
    return VerificationResult(False, selected, candidates, reason)
