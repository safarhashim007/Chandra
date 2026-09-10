"""Metadata contract and sidecar parsing; raster readers are deliberately optional."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .lunar_crs import normalize_east_longitude, validate_latitude
from .pixel_world import AffineTransform


@dataclass(frozen=True)
class LunarImageMetadata:
    product_id: str
    source_path: str
    width: int
    height: int
    center_lat: float
    center_lon_east: float
    gsd_m_per_px: float
    transform: AffineTransform
    acquisition_time: str | None = None
    projection: str | None = None
    lunar_datum: str | None = None
    incidence_angle: float | None = None
    emission_angle: float | None = None
    phase_angle: float | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image width and height must be positive")
        if self.gsd_m_per_px <= 0:
            raise ValueError("GSD must be positive")
        validate_latitude(self.center_lat)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def metadata_from_mapping(data: dict[str, Any]) -> LunarImageMetadata:
    transform = data.get("transform")
    if isinstance(transform, dict):
        transform = AffineTransform(**transform)
    if not isinstance(transform, AffineTransform):
        raise ValueError("metadata transform must be an AffineTransform or six-field mapping")
    longitude = data.get("center_lon_east", data.get("center_lon"))
    if longitude is None:
        raise ValueError("metadata must include center_lon_east")
    return LunarImageMetadata(
        product_id=str(data["product_id"]),
        source_path=str(data["source_path"]),
        width=int(data["width"]),
        height=int(data["height"]),
        center_lat=float(data["center_lat"]),
        center_lon_east=normalize_east_longitude(float(longitude)),
        gsd_m_per_px=float(data["gsd_m_per_px"]),
        transform=transform,
        acquisition_time=data.get("acquisition_time"),
        projection=data.get("projection"),
        lunar_datum=data.get("lunar_datum"),
        incidence_angle=data.get("incidence_angle"),
        emission_angle=data.get("emission_angle"),
        phase_angle=data.get("phase_angle"),
    )


def read_metadata_sidecar(path: str | Path) -> LunarImageMetadata:
    with Path(path).open(encoding="utf-8") as handle:
        return metadata_from_mapping(json.load(handle))
