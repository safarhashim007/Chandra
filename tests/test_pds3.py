from pathlib import Path

import numpy as np
import pytest

from chandrappan.data.pds3 import parse_pds3_header, read_pds3_header, sampled_band_statistics


def _header() -> bytes:
    return b"""PDS_VERSION_ID = PDS3
RECORD_BYTES = 1024
LABEL_RECORDS = 1
^IMAGE = 2
PRODUCT_ID = TEST_PRODUCT
START_TIME = 2012-01-01T00:00:00
STOP_TIME = 2012-01-01T00:01:00
MAP_SCALE = 5.0 <METERS/PIXEL>
MAXIMUM_LATITUDE = 1.0 <DEG>
MINIMUM_LATITUDE = -1.0 <DEG>
EASTERNMOST_LONGITUDE = 2.0 <DEG>
WESTERNMOST_LONGITUDE = 0.0 <DEG>
DESCRIPTION = "four band map-projected test product"
OBJECT = IMAGE
LINES = 2
LINE_SAMPLES = 3
BANDS = 4
END_OBJECT = IMAGE
"""


def test_reads_strict_header_and_sampled_backplanes(tmp_path: Path) -> None:
    source = tmp_path / "test.img"
    label = _header().ljust(1024, b" ")
    raster = np.arange(24, dtype="<f4").reshape(4, 2, 3)
    raster[1, 0, 0] = np.float32(-3.4028227e38)
    source.write_bytes(label + raster.tobytes())

    header = read_pds3_header(source)
    stats = sampled_band_statistics(source, header, stride=1)

    assert header.product_id == "TEST_PRODUCT"
    assert header.data_offset_bytes == 1024
    assert header.expected_raster_bytes == 96
    assert stats["phase_deg"]["median"] == pytest.approx(9.0)
    assert stats["incidence_deg"]["max"] == pytest.approx(23.0)


def test_rejects_truncated_raster(tmp_path: Path) -> None:
    source = tmp_path / "truncated.img"
    source.write_bytes(_header().ljust(1024, b" ") + b"x")
    with pytest.raises(ValueError, match="byte count mismatch"):
        read_pds3_header(source)


def test_rejects_partial_label() -> None:
    with pytest.raises(ValueError, match="complete"):
        parse_pds3_header("PDS_VERSION_ID = PDS3")
