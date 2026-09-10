"""Canonical, validated image manifests for map-projected lunar imagery."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shapely import wkt
from shapely.geometry import Polygon

from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.geo.footprint import footprint_corners_world
from chandrappan.geo.lunar_crs import normalize_east_longitude


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: str
    source: str
    product_id: str
    instrument: str | None
    spacecraft: str | None
    footprint_wkt: str
    min_longitude: float
    max_longitude: float
    min_latitude: float
    max_latitude: float
    acquisition_time: str | None
    incidence_angle: float | None
    emission_angle: float | None
    phase_angle: float | None
    gsd_m_per_px: float
    width: int
    height: int
    nodata: float | None
    region_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metadata value must be finite, got {value!r}")
    return result


def image_record_from_lroc(
    raster_path: str | Path,
    label_path: str | Path | None = None,
    *,
    region_id: str | None = None,
) -> ImageRecord:
    """Create a manifest row from real LROC metadata without filling unknowns."""
    raster = Path(raster_path)
    metadata = read_lroc_metadata(raster, label_path)
    try:
        import rasterio
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise RuntimeError("manifest creation requires rasterio and pyproj") from exc

    with rasterio.open(raster) as dataset:
        crs = CRS.from_wkt(dataset.crs.to_wkt())
        transformer = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
        world = footprint_corners_world(metadata)
        longitudes, latitudes = transformer.transform(
            [point[0] for point in world], [point[1] for point in world]
        )
        longitudes = [normalize_east_longitude(value) for value in longitudes]
        footprint = Polygon(zip(longitudes, latitudes, strict=True))
        nodata = _finite_or_none(dataset.nodata)

    if not footprint.is_valid or footprint.is_empty:
        raise ValueError(f"invalid LROC footprint for {raster}")
    derived_region = region_id or (
        f"lat{math.floor(metadata.center_lat / 5) * 5:03d}_"
        f"lon{math.floor(metadata.center_lon_east / 5) * 5:03d}"
    )
    instrument = next(
        (value for value in ("NAC", "WAC") if value in metadata.product_id.upper()), None
    )
    return ImageRecord(
        image_id=metadata.product_id,
        image_path=str(raster),
        source="LROC",
        product_id=metadata.product_id,
        instrument=instrument,
        spacecraft="LRO",
        footprint_wkt=footprint.wkt,
        min_longitude=min(longitudes),
        max_longitude=max(longitudes),
        min_latitude=min(latitudes),
        max_latitude=max(latitudes),
        acquisition_time=metadata.acquisition_time,
        incidence_angle=metadata.incidence_angle,
        emission_angle=metadata.emission_angle,
        phase_angle=metadata.phase_angle,
        gsd_m_per_px=metadata.gsd_m_per_px,
        width=metadata.width,
        height=metadata.height,
        nodata=nodata,
        region_id=derived_region,
    )


def validate_manifest(
    records: Iterable[ImageRecord], *, check_paths: bool = True
) -> list[ImageRecord]:
    rows = list(records)
    seen_ids: set[str] = set()
    seen_products: set[str] = set()
    for row in rows:
        if row.image_id in seen_ids or row.product_id in seen_products:
            raise ValueError(f"duplicate image/product ID: {row.image_id}/{row.product_id}")
        seen_ids.add(row.image_id)
        seen_products.add(row.product_id)
        if check_paths and not Path(row.image_path).exists():
            raise FileNotFoundError(row.image_path)
        if row.width <= 0 or row.height <= 0 or row.gsd_m_per_px <= 0:
            raise ValueError(f"invalid dimensions or GSD for {row.image_id}")
        if not all(
            math.isfinite(value)
            for value in (row.min_longitude, row.max_longitude, row.min_latitude, row.max_latitude)
        ):
            raise ValueError(f"non-finite bounds for {row.image_id}")
        geometry = wkt.loads(row.footprint_wkt)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"invalid footprint for {row.image_id}")
        for value in (row.incidence_angle, row.emission_angle, row.phase_angle, row.nodata):
            _finite_or_none(value)
    return rows


def write_manifest(records: Iterable[ImageRecord], path: str | Path) -> None:
    rows = [row.as_dict() for row in validate_manifest(records, check_paths=False)]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet manifests require pyarrow") from exc
        pq.write_table(pa.Table.from_pylist(rows), destination)
    elif destination.suffix == ".json":
        destination.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    elif destination.suffix == ".csv":
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError(f"unsupported manifest format: {destination.suffix}")


def read_manifest(path: str | Path) -> list[ImageRecord]:
    source = Path(path)
    if source.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(source).to_pylist()
    elif source.suffix == ".json":
        rows = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        numeric = {
            "min_longitude",
            "max_longitude",
            "min_latitude",
            "max_latitude",
            "incidence_angle",
            "emission_angle",
            "phase_angle",
            "gsd_m_per_px",
            "width",
            "height",
            "nodata",
        }
        for row in rows:
            for field in numeric:
                if row[field] == "":
                    row[field] = None
                elif field in {"width", "height"}:
                    row[field] = int(row[field])
                else:
                    row[field] = float(row[field])
    else:
        raise ValueError(f"unsupported manifest format: {source.suffix}")
    return validate_manifest([ImageRecord(**row) for row in rows])
