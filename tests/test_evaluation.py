import numpy as np
import pytest
import torch

from chandrappan.evaluation.analysis import confidence_subsets, error_distribution
from chandrappan.evaluation.metrics import correspondence_metrics
from chandrappan.evaluation.protocol import AcceptanceConfig, accept_registration, vrr_far
from chandrappan.evaluation.quality_gate import (
    QualityGateConfig,
    evaluate_quality_gate,
    require_quality_gate,
)
from chandrappan.evaluation.thresholds import select_validation_thresholds


def test_correspondence_metrics_and_pck() -> None:
    target = torch.zeros(1, 2, 2, 2)
    predicted = target.clone()
    predicted[..., 0] = 1
    result = correspondence_metrics(predicted, target, torch.ones(1, 2, 2, dtype=torch.bool))
    assert result["median_epe_px"] == pytest.approx(1)
    assert result["pck_1"] == pytest.approx(1)
    assert result["pck_3"] == pytest.approx(1)


def test_acceptance_and_vrr_far_are_deterministic() -> None:
    config = AcceptanceConfig(min_correspondences=10, min_inliers=5)
    metrics = {
        "correspondences": 20,
        "inliers": 10,
        "inlier_ratio": 0.5,
        "reprojection_error_px": 1,
        "spatial_coverage": 0.5,
        "grid_coverage": 0.5,
        "hull_coverage": 0.5,
        "scale": 1,
        "rotation_degrees": 1,
        "shear": 0,
        "anisotropy": 1,
    }
    assert accept_registration(metrics, config)
    assert vrr_far([(True, True), (True, False), (False, True), (False, False)]) == {
        "positive_count": 2,
        "negative_count": 2,
        "vrr": 0.5,
        "far": 0.5,
    }


def test_quality_gate_pass_and_fail() -> None:
    before = {"median_epe_px": 2, "pck_1": 0.4, "pck_3": 0.6, "pck_5": 0.7, "vrr": 0.5, "far": 0.1}
    after = {"median_epe_px": 1.5, "pck_1": 0.4, "pck_3": 0.7, "pck_5": 0.7, "vrr": 0.5, "far": 0.1}
    result = evaluate_quality_gate(before, after, QualityGateConfig())
    assert result["passed"]
    require_quality_gate(result)
    failed = dict(after, median_epe_px=3.0, pck_1=0.1)
    result = evaluate_quality_gate(before, failed, QualityGateConfig())
    assert not result["passed"]
    with pytest.raises(RuntimeError, match="FULL TRAINING BLOCKED"):
        require_quality_gate(result)


def test_error_distribution_and_confidence_analysis() -> None:
    distribution = error_distribution(np.array([0.5, 1, 2, 10, 100]))
    assert distribution["median"] == pytest.approx(2)
    assert distribution["fraction_above_10_px"] == pytest.approx(0.2)
    subsets = confidence_subsets(np.array([1, 10, 2, 20]), np.array([0.9, 0.1, 0.8, 0.7]))
    assert subsets[0]["median_epe_px"] == pytest.approx(6)
    assert subsets[-1]["count"] == 1


def test_threshold_selection_is_validation_only() -> None:
    base = {
        "correspondences": 20,
        "inliers": 10,
        "inlier_ratio": 0.5,
        "reprojection_error_px": 1,
        "spatial_coverage": 0.5,
        "grid_coverage": 0.5,
        "hull_coverage": 0.5,
        "scale": 1,
        "rotation_degrees": 1,
        "shear": 0,
        "anisotropy": 1,
    }
    rows = [
        {"is_positive": True, "metrics": base},
        {"is_positive": False, "metrics": dict(base, inliers=0)},
    ]
    result = select_validation_thresholds(
        rows,
        [AcceptanceConfig(min_correspondences=10, min_inliers=5)],
        far_limit=0,
        dataset_name="validation",
    )
    assert result.vrr == 1
    with pytest.raises(ValueError, match="T0"):
        select_validation_thresholds(rows, [AcceptanceConfig()], far_limit=0, dataset_name="T0_v1")
