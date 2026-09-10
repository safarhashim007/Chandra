"""Minimal raster-backed LROC metadata adapter."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree

from chandrappan.geo.lunar_crs import (
    MOON_MEAN_RADIUS_M,
    normalize_east_longitude,
    validate_latitude,
)
from chandrappan.geo.metadata import LunarImageMetadata
from chandrappan.geo.pixel_world import AffineTransform


def read_lroc_metadata(
    raster_path: str | Path, label_path: str | Path | None = None
) -> LunarImageMetadata:
    """Read a map-projected LROC raster and preserve its Moon-specific georeferencing."""
    try:
        import rasterio
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise RuntimeError("LROC ingestion requires rasterio and pyproj") from exc

    raster = Path(raster_path)
    with rasterio.open(raster) as dataset:
        if dataset.crs is None:
            raise ValueError(f"LROC raster has no CRS: {raster}")
        crs = CRS.from_wkt(dataset.crs.to_wkt())
        radius = crs.ellipsoid.semi_major_metre
        if radius is None or abs(radius - MOON_MEAN_RADIUS_M) > 1.0:
            raise ValueError(f"unsupported lunar radius {radius!r}; expected {MOON_MEAN_RADIUS_M}")
        transform = AffineTransform(
            dataset.transform.c,
            dataset.transform.a,
            dataset.transform.b,
            dataset.transform.f,
            dataset.transform.d,
            dataset.transform.e,
        )
        tags = dataset.tags()
        width, height = dataset.width, dataset.height
        center_x, center_y = transform.pixel_to_world(width / 2, height / 2)
        to_geographic = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
        center_lon, center_lat = to_geographic.transform(center_x, center_y)
        product_id = tags.get("PRODUCT_ID") or raster.stem
        gsd = (transform.x_col**2 + transform.y_col**2) ** 0.5

    label = _read_label(label_path) if label_path else {}
    center_lat = float(label.get("center_lat", center_lat))
    center_lon = float(label.get("center_lon", center_lon))
    validate_latitude(center_lat)
    return LunarImageMetadata(
        product_id=product_id,
        source_path=str(raster),
        width=width,
        height=height,
        center_lat=center_lat,
        center_lon_east=normalize_east_longitude(center_lon),
        gsd_m_per_px=gsd,
        transform=transform,
        projection=crs.name,
        lunar_datum=crs.datum.name if crs.datum else None,
        acquisition_time=tags.get("PRODUCT_CREATION_TIME"),
    )


def _read_label(path: str | Path) -> dict[str, float]:
    root = ElementTree.parse(path).getroot()
    text = " ".join(value.strip() for value in root.itertext() if value.strip())
    match = re.search(r"center of this tile is at\s+([0-9.]+)([NS]),\s*([0-9.]+)([EW])", text)
    if not match:
        return {}
    lat, lat_dir, lon, lon_dir = match.groups()
    return {
        "center_lat": float(lat) * (1 if lat_dir == "N" else -1),
        "center_lon": float(lon) * (1 if lon_dir == "E" else -1),
    }
