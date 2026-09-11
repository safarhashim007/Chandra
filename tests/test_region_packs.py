import numpy as np
import pytest

from chandrappan.region_packs import fractional_zncc_peak, intensity_descriptor, rank_references


def test_descriptor_ranking_is_stable_and_uses_all_references() -> None:
    query = intensity_descriptor(np.arange(100, dtype=np.float32).reshape(10, 10))
    alternate = intensity_descriptor(np.concatenate((np.zeros(50), np.ones(50))).reshape(10, 10))
    ranked = rank_references(query, {"B": alternate, "A": query})
    assert [candidate.product_id for candidate in ranked] == ["A", "B"]
    assert ranked[0].score == pytest.approx(1.0)


def test_fractional_zncc_peak_estimates_subpixel_offset() -> None:
    y, x = np.indices((7, 7), dtype=np.float64)
    correlation = 1.0 - (x - 3.25) ** 2 * 0.1 - (y - 2.75) ** 2 * 0.1
    refined = fractional_zncc_peak(correlation)
    assert refined.accepted
    assert refined.dx_px == pytest.approx(0.25)
    assert refined.dy_px == pytest.approx(-0.25)


def test_fractional_zncc_peak_rejects_ambiguous_maxima() -> None:
    correlation = np.zeros((5, 5))
    correlation[2, 2] = 0.9
    correlation[0, 0] = 0.89
    assert fractional_zncc_peak(correlation).rejection_reason == "ambiguous_zncc_peak"


def test_fractional_zncc_peak_includes_integer_search_displacement() -> None:
    y, x = np.indices((7, 7), dtype=np.float64)
    correlation = 1.0 - (x - 4.0) ** 2 * 0.1 - (y - 2.0) ** 2 * 0.1
    refined = fractional_zncc_peak(correlation)
    assert refined.accepted
    assert refined.dx_px == pytest.approx(1.0)
    assert refined.dy_px == pytest.approx(-1.0)
