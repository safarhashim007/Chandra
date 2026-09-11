import numpy as np

from chandrappan.rp001_demo import PREPROCESSING_METHODS, preprocess_science
from scripts.build_rp001_demo_data import _window_mean


def test_preprocessing_is_finite_deterministic_and_geometry_neutral() -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    for method in PREPROCESSING_METHODS:
        first = preprocess_science(image, method)
        second = preprocess_science(image, method)
        assert first.shape == image.shape
        assert np.isfinite(first).all()
        assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0
        assert np.array_equal(first, second)


def test_window_mean_reports_nodata_aware_coverage() -> None:
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    coverage = _window_mean(mask, 2)
    assert coverage.shape == (3, 3)
    assert coverage[0, 0] == 0.75
    assert coverage[1, 1] == 1.0
