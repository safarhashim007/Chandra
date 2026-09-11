import pytest

from chandrappan.data.curator import curate_train_pairs
from chandrappan.data.manifest import ImageRecord
from chandrappan.data.pairs import PairRecord
from chandrappan.data.scientific import DenseGTQuality


def _image(product_id: str, region_id: str) -> ImageRecord:
    return ImageRecord(
        image_id=product_id,
        image_path=f"/tmp/{product_id}.TIF",
        source="LROC",
        product_id=product_id,
        instrument="NAC",
        spacecraft="LRO",
        footprint_wkt="POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        min_longitude=0,
        max_longitude=1,
        min_latitude=0,
        max_latitude=1,
        acquisition_time=None,
        incidence_angle=None,
        emission_angle=None,
        phase_angle=None,
        gsd_m_per_px=1,
        width=10,
        height=10,
        nodata=None,
        region_id=region_id,
    )


def _pair(pair_id: str, image_a: str, image_b: str, region_id: str) -> PairRecord:
    return PairRecord(
        pair_id=pair_id,
        image_a=image_a,
        image_b=image_b,
        region_id=region_id,
        overlap_area=1,
        overlap_ratio=0.5,
        gsd_a_m_per_px=1,
        gsd_b_m_per_px=1,
        relative_gsd_ratio=1,
        acquisition_time_delta_seconds=None,
        incidence_angle_delta=None,
        emission_angle_delta=None,
        phase_angle_delta=None,
    )


def test_curator_uses_all_valid_pairs_with_normalized_sampling_weights() -> None:
    records = [
        _image("NAC_PHO_E001N0001_M1L", "E001N0001"),
        _image("NAC_PHO_E001N0001_M2L", "E001N0001"),
        _image("NAC_PHO_E002N0002_M1L", "E002N0002"),
        _image("NAC_PHO_E002N0002_M2L", "E002N0002"),
    ]
    pairs = [
        _pair(
            "NAC_PHO_E001N0001_M1L__NAC_PHO_E001N0001_M2L",
            "NAC_PHO_E001N0001_M1L",
            "NAC_PHO_E001N0001_M2L",
            "E001N0001",
        ),
        _pair(
            "NAC_PHO_E002N0002_M1L__NAC_PHO_E002N0002_M2L",
            "NAC_PHO_E002N0002_M1L",
            "NAC_PHO_E002N0002_M2L",
            "E002N0002",
        ),
    ]
    quality = {
        pair.pair_id: DenseGTQuality(0.2, 0.0, 0.0, 0.0, 0.0, 100, True, None) for pair in pairs
    }

    curated = curate_train_pairs(records, pairs, quality, max_pairs_per_region=0)

    assert sum(row.selected for row in curated) == 2
    assert sum(row.sampling_weight for row in curated) == pytest.approx(1.0)
    assert {row.difficulty_class for row in curated} == {"medium"}
