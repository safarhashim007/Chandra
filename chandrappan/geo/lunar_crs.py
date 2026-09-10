"""Canonical lunar CRS definitions and explicit coordinate validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians

MOON_MEAN_RADIUS_M = 1_737_400.0
LONGITUDE_CONVENTION = "east_positive"
LATITUDE_CONVENTION = "planetocentric"


@dataclass(frozen=True)
class LunarCoordinate:
    """A planetocentric coordinate in the canonical east-positive convention."""

    latitude_deg: float
    longitude_deg_east: float

    def normalized(self) -> LunarCoordinate:
        return LunarCoordinate(self.latitude_deg, self.longitude_deg_east % 360.0)


def validate_latitude(latitude_deg: float) -> float:
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError(f"Planetocentric latitude must be in [-90, 90], got {latitude_deg}")
    return latitude_deg


def normalize_east_longitude(longitude_deg: float) -> float:
    """Normalize an east-positive longitude to [0, 360)."""
    return longitude_deg % 360.0


def west_to_east_longitude(longitude_deg_west: float) -> float:
    return normalize_east_longitude(-longitude_deg_west)


def east_to_west_longitude(longitude_deg_east: float) -> float:
    return normalize_east_longitude(-longitude_deg_east)


def local_meters_per_degree(latitude_deg: float) -> tuple[float, float]:
    """Return (northing, easting) metres per degree on the mean-radius sphere."""
    validate_latitude(latitude_deg)
    per_degree = MOON_MEAN_RADIUS_M * 3.141592653589793 / 180.0
    return per_degree, per_degree * cos(radians(latitude_deg))
