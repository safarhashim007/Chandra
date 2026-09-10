from pathlib import Path

import pytest

from chandrappan.data.manifest import ImageRecord, read_manifest, validate_manifest, write_manifest
from chandrappan.data.pairs import generate_pairs
from chandrappan.data.splits import geographic_split, validate_no_leakage


def _record(image_id: str, region_id: str, x: float, *, path: str = "missing.tif") -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        image_path=path,
        source="test",
        product_id=image_id,
        instrument=None,
        spacecraft=None,
        footprint_wkt=f"POLYGON (({x} 0, {x + 10} 0, {x + 10} 10, {x} 10, {x} 0))",
        min_longitude=x,
        max_longitude=x + 10,
        min_latitude=0,
        max_latitude=10,
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


def test_manifest_roundtrip_and_missing_metadata(tmp_path: Path) -> None:
    records = [_record("a", "region-a", 0, path=str(Path(__file__)))]
    validate_manifest(records, check_paths=False)
    path = tmp_path / "images.parquet"
    write_manifest(records, path)
    restored = read_manifest(path)
    assert restored == records
    assert restored[0].incidence_angle is None


def test_manifest_rejects_duplicate_products() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(
            [_record("a", "region-a", 0), _record("a", "region-b", 20)], check_paths=False
        )


def test_pairs_keep_only_geographic_overlap() -> None:
    records = [
        _record("a", "region-a", 0),
        _record("b", "region-a", 5),
        _record("c", "region-c", 100),
    ]
    pairs = generate_pairs(records, min_overlap_ratio=0.1)
    assert [pair.pair_id for pair in pairs] == ["a__b"]
    assert pairs[0].relative_gsd_ratio == 1


def test_geographic_split_is_deterministic_and_region_safe() -> None:
    records = [_record("a", "a", 0), _record("b", "b", 20), _record("c", "c", 40)]
    first = geographic_split(records, seed=7)
    second = geographic_split(records, seed=7)
    assert first == second
    validate_no_leakage(first)


def test_leakage_rejects_cross_split_region_reuse() -> None:
    with pytest.raises(ValueError, match="region reused"):
        validate_no_leakage(
            {"train": [_record("a", "same", 0)], "test": [_record("b", "same", 20)]}
        )
