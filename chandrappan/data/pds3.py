"""Small, strict readers for the PDS3 SDPPHO image headers used by Region Packs.

The official SDPPHO ``.IMG`` files carry an embedded PDS3 label followed by a
four-band, band-sequential float raster.  The separate GeoTIFF is a convenient
map-projected browse representation, but is only the science (I/F) band.  This
module deliberately reads the original image for illumination backplanes and
does not synthesise acquisition metadata.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Pds3ImageHeader:
    """The subset of a map-projected SDPPHO PDS3 label needed by RP builds."""

    product_id: str
    start_time: str
    stop_time: str | None
    record_bytes: int
    label_records: int
    image_record: int
    lines: int
    line_samples: int
    bands: int
    map_scale_m_per_px: float
    maximum_latitude: float
    minimum_latitude: float
    easternmost_longitude: float
    westernmost_longitude: float
    description: str

    @property
    def data_offset_bytes(self) -> int:
        return (self.image_record - 1) * self.record_bytes

    @property
    def expected_raster_bytes(self) -> int:
        return self.lines * self.line_samples * self.bands * np.dtype("<f4").itemsize

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _value(label: str, key: str, *, required: bool = True) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", label, re.MULTILINE)
    if match is None:
        if required:
            raise ValueError(f"PDS3 label missing {key}")
        return None
    value = match.group(1).strip()
    # Values with units are written as e.g. ``5.0 <METERS/PIXEL>``.
    return re.sub(r"\s*<[^>]+>\s*$", "", value).strip().strip('"')


def parse_pds3_header(value: bytes | str) -> Pds3ImageHeader:
    """Parse a PDS3 SDPPHO header without accepting a partial label."""
    # ``read_pds3_header`` deliberately reads a generous prefix.  It therefore
    # contains binary image bytes after the fixed-length label record.
    label = value.decode("ascii", "replace") if isinstance(value, bytes) else value
    if "PDS_VERSION_ID" not in label or "END_OBJECT = IMAGE" not in label:
        raise ValueError("not a complete PDS3 SDPPHO image label")
    return Pds3ImageHeader(
        product_id=str(_value(label, "PRODUCT_ID")),
        start_time=str(_value(label, "START_TIME")),
        stop_time=_value(label, "STOP_TIME", required=False),
        record_bytes=int(str(_value(label, "RECORD_BYTES"))),
        label_records=int(str(_value(label, "LABEL_RECORDS"))),
        image_record=int(str(_value(label, "^IMAGE"))),
        lines=int(str(_value(label, "LINES"))),
        line_samples=int(str(_value(label, "LINE_SAMPLES"))),
        bands=int(str(_value(label, "BANDS"))),
        map_scale_m_per_px=float(str(_value(label, "MAP_SCALE"))),
        maximum_latitude=float(str(_value(label, "MAXIMUM_LATITUDE"))),
        minimum_latitude=float(str(_value(label, "MINIMUM_LATITUDE"))),
        easternmost_longitude=float(str(_value(label, "EASTERNMOST_LONGITUDE"))),
        westernmost_longitude=float(str(_value(label, "WESTERNMOST_LONGITUDE"))),
        description=str(_value(label, "DESCRIPTION")),
    )


def read_pds3_header(path: str | Path) -> Pds3ImageHeader:
    """Read a local SDPPHO header and reject truncated labels or rasters."""
    source = Path(path)
    # SDPPHO labels fit in their first fixed record; retaining 64 KiB makes the
    # error clear for a malformed future product without reading raster data.
    with source.open("rb") as handle:
        prefix = handle.read(65536)
    header = parse_pds3_header(prefix)
    expected = header.data_offset_bytes + header.expected_raster_bytes
    actual = source.stat().st_size
    if actual != expected:
        raise ValueError(
            f"PDS3 SDPPHO byte count mismatch for {source}: expected {expected}, got {actual}"
        )
    if header.bands != 4:
        raise ValueError(f"expected four SDPPHO bands, got {header.bands} for {source}")
    return header


def sampled_band_statistics(
    path: str | Path, header: Pds3ImageHeader, *, stride: int = 64
) -> dict[str, dict[str, float | int]]:
    """Compute robust original-pixel statistics for I/F and angle backplanes.

    Sampling is deterministic and avoids treating floating PDS special values
    as observations.  Statistics are descriptive catalog metadata, not ground
    truth for a single pixel or a substitute for photometric calibration.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    raster = np.memmap(
        Path(path),
        dtype="<f4",
        mode="r",
        offset=header.data_offset_bytes,
        shape=(header.bands, header.lines, header.line_samples),
    )
    names = ("iof", "phase_deg", "emission_deg", "incidence_deg")
    result: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(names):
        values = np.asarray(raster[index, ::stride, ::stride], dtype=np.float64)
        # PDS3 SDPPHO special constants are IEEE float sentinels near the
        # representable extrema (for example -3.4028227e38), not NaN.  They
        # must not become physically absurd illumination observations.
        valid = values[np.isfinite(values) & (np.abs(values) < 1.0e30)]
        if name != "iof":
            valid = valid[(valid >= 0.0) & (valid <= 360.0)]
        if valid.size == 0:
            raise ValueError(f"no finite {name} samples in {path}")
        result[name] = {
            "sample_count": int(valid.size),
            "min": float(np.min(valid)),
            "p05": float(np.quantile(valid, 0.05)),
            "median": float(np.median(valid)),
            "p95": float(np.quantile(valid, 0.95)),
            "max": float(np.max(valid)),
        }
    return result
