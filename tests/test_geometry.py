import math
from dataclasses import replace

import numpy as np
import pytest

from chandrappan.evaluation.visualize import save_match_diagnostic
from chandrappan.geometry.coverage import hull_coverage, spatial_coverage
from chandrappan.geometry.verification import VerificationConfig, verify_registration


def _grid() -> np.ndarray:
    x, y = np.meshgrid(np.linspace(20, 180, 8), np.linspace(20, 180, 8))
    return np.column_stack((x.ravel(), y.ravel())).astype(np.float32)


def test_coverage_distinguishes_clustered_and_distributed_points() -> None:
    clustered = np.array([[50, 50], [55, 50], [50, 55], [55, 55]], dtype=float)
    distributed = _grid()
    assert spatial_coverage(clustered, (200, 200))["grid_coverage"] < 0.05
    assert spatial_coverage(distributed, (200, 200))["grid_coverage"] > 0.5
    assert hull_coverage(clustered, (200, 200)) < 0.01
    assert hull_coverage(distributed, (200, 200)) > 0.5


def _config() -> VerificationConfig:
    return VerificationConfig(
        min_matches=8,
        min_inliers=30,
        min_inlier_ratio=0.6,
        min_grid_coverage=0.3,
        min_hull_coverage=0.3,
        max_rotation_degrees=30,
        max_scale=2,
    )


def test_similarity_recovery_rejects_outliers() -> None:
    source = _grid()
    angle = math.radians(8)
    linear = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    target = source @ linear.T + np.array([12, -4])
    target[:8] += 80
    result = verify_registration(source, target, (200, 200), config=_config())
    assert result.accepted
    assert result.selected is not None
    assert result.selected.model_type == "similarity"
    assert result.selected.inliers >= 50
    assert result.selected.rotation_degrees == pytest.approx(8, abs=1)


def test_affine_and_homography_models_are_supported() -> None:
    source = _grid()
    affine = np.array([[1.05, 0.08, 4], [-0.03, 0.97, 8]], dtype=float)
    target = np.column_stack((source, np.ones(len(source)))) @ affine.T
    affine_result = verify_registration(
        source, target, (200, 200), config=replace(_config(), model_order=("affine",))
    )
    assert affine_result.accepted
    assert affine_result.selected is not None
    assert affine_result.selected.model_type == "affine"

    homography = np.array([[1, 0.02, 3], [0.01, 1, 4], [0.0002, -0.0001, 1]], dtype=float)
    homogeneous = np.column_stack((source, np.ones(len(source)))) @ homography.T
    target = homogeneous[:, :2] / homogeneous[:, 2:3]
    homography_result = verify_registration(
        source, target, (200, 200), config=replace(_config(), model_order=("homography",))
    )
    assert homography_result.accepted
    assert homography_result.selected is not None
    assert homography_result.selected.model_type == "homography"


def test_degenerate_matches_are_rejected() -> None:
    source = np.column_stack((np.linspace(10, 180, 40), np.full(40, 50))).astype(np.float32)
    target = source + np.array([5, 2], dtype=np.float32)
    result = verify_registration(source, target, (200, 200), config=_config())
    assert not result.accepted


def test_match_diagnostic_smoke(tmp_path) -> None:
    image = np.zeros((20, 20), dtype=float)
    points = np.array([[2, 2], [10, 10], [15, 4]], dtype=float)
    output = tmp_path / "diagnostic.png"
    save_match_diagnostic(image, image, points, points, np.array([True, False, True]), output)
    assert output.exists()
