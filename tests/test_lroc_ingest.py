from pathlib import Path

import pytest

from chandrappan.data.lroc_ingest import read_lroc_metadata


def test_real_lroc_fixture_metadata_and_roundtrip() -> None:
    raster = Path("data/raw/lroc_fixture/WAC_TIO2_E350N0450.TIF")
    label = raster.with_suffix(".xml")
    if not raster.exists() or not label.exists():
        pytest.skip("ignored LROC fixture is not present")
    metadata = read_lroc_metadata(raster, label)
    x = [0.0, metadata.width / 2, metadata.width - 1.0, 1234.25]
    y = [0.0, metadata.height / 2, metadata.height - 1.0, 4321.75]
    world = metadata.transform.pixel_to_world(x, y)
    recovered = metadata.transform.world_to_pixel(*world)
    assert (
        max((a - b) ** 2 + (c - d) ** 2 for a, b, c, d in zip(x, recovered[0], y, recovered[1]))
        ** 0.5
        < 1e-9
    )
    assert metadata.product_id == "WAC_TIO2_E350N0450"
    assert metadata.center_lat == pytest.approx(35.0)
    assert metadata.center_lon_east == pytest.approx(45.0)
    assert metadata.gsd_m_per_px == pytest.approx(398.991452949)
