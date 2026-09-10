"""Geographic candidate-pair generation from canonical image manifests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import combinations
from typing import Any

from shapely import wkt

from .manifest import ImageRecord


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    image_a: str
    image_b: str
    region_id: str
    overlap_area: float
    overlap_ratio: float
    gsd_a_m_per_px: float
    gsd_b_m_per_px: float
    relative_gsd_ratio: float
    acquisition_time_delta_seconds: float | None
    incidence_angle_delta: float | None
    emission_angle_delta: float | None
    phase_angle_delta: float | None
    label: str = "positive"
    negative_type: str | None = None
    geographic_separation_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _difference(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else abs(first - second)


def _time_difference(first: str | None, second: str | None) -> float | None:
    if first is None or second is None:
        return None
    try:
        return abs((datetime.fromisoformat(first) - datetime.fromisoformat(second)).total_seconds())
    except ValueError:
        return None


def generate_pairs(
    records: Iterable[ImageRecord], *, min_overlap_ratio: float = 0.01
) -> list[PairRecord]:
    rows = list(records)
    result: list[PairRecord] = []
    for first, second in combinations(rows, 2):
        geometry_a = wkt.loads(first.footprint_wkt)
        geometry_b = wkt.loads(second.footprint_wkt)
        overlap_area = geometry_a.intersection(geometry_b).area
        denominator = min(geometry_a.area, geometry_b.area)
        overlap_ratio = overlap_area / denominator if denominator else 0.0
        if overlap_ratio < min_overlap_ratio:
            continue
        result.append(
            PairRecord(
                pair_id=f"{first.image_id}__{second.image_id}",
                image_a=first.image_id,
                image_b=second.image_id,
                region_id=first.region_id if first.region_id == second.region_id else "mixed",
                overlap_area=overlap_area,
                overlap_ratio=overlap_ratio,
                gsd_a_m_per_px=first.gsd_m_per_px,
                gsd_b_m_per_px=second.gsd_m_per_px,
                relative_gsd_ratio=max(first.gsd_m_per_px, second.gsd_m_per_px)
                / min(first.gsd_m_per_px, second.gsd_m_per_px),
                acquisition_time_delta_seconds=_time_difference(
                    first.acquisition_time, second.acquisition_time
                ),
                incidence_angle_delta=_difference(first.incidence_angle, second.incidence_angle),
                emission_angle_delta=_difference(first.emission_angle, second.emission_angle),
                phase_angle_delta=_difference(first.phase_angle, second.phase_angle),
            )
        )
    return result


def _separation_meters(first: ImageRecord, second: ImageRecord) -> float:
    """Approximate centroid separation for negative-pair diagnostics."""
    import math

    lat1 = math.radians((first.min_latitude + first.max_latitude) / 2)
    lat2 = math.radians((second.min_latitude + second.max_latitude) / 2)
    lon1 = math.radians((first.min_longitude + first.max_longitude) / 2)
    lon2 = math.radians((second.min_longitude + second.max_longitude) / 2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    haversine = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 1_737_400.0 * 2 * math.asin(min(1.0, math.sqrt(haversine)))


def generate_negative_pairs(
    records: Iterable[ImageRecord], *, max_overlap_ratio: float = 0.0
) -> list[PairRecord]:
    """Generate easy geographic negatives; no visual similarity is inferred."""
    rows = list(records)
    result: list[PairRecord] = []
    for first, second in combinations(rows, 2):
        geometry_a = wkt.loads(first.footprint_wkt)
        geometry_b = wkt.loads(second.footprint_wkt)
        overlap_area = geometry_a.intersection(geometry_b).area
        denominator = min(geometry_a.area, geometry_b.area)
        overlap_ratio = overlap_area / denominator if denominator else 0.0
        if overlap_ratio > max_overlap_ratio:
            continue
        result.append(
            PairRecord(
                pair_id=f"{first.image_id}__{second.image_id}",
                image_a=first.image_id,
                image_b=second.image_id,
                region_id="negative",
                overlap_area=overlap_area,
                overlap_ratio=overlap_ratio,
                gsd_a_m_per_px=first.gsd_m_per_px,
                gsd_b_m_per_px=second.gsd_m_per_px,
                relative_gsd_ratio=max(first.gsd_m_per_px, second.gsd_m_per_px)
                / min(first.gsd_m_per_px, second.gsd_m_per_px),
                acquisition_time_delta_seconds=_time_difference(
                    first.acquisition_time, second.acquisition_time
                ),
                incidence_angle_delta=_difference(first.incidence_angle, second.incidence_angle),
                emission_angle_delta=_difference(first.emission_angle, second.emission_angle),
                phase_angle_delta=_difference(first.phase_angle, second.phase_angle),
                label="negative",
                negative_type="easy_geographic",
                geographic_separation_m=_separation_meters(first, second),
            )
        )
    return result


def generate_explicit_negative_pairs(
    records: Iterable[ImageRecord],
    image_pairs: Iterable[tuple[str, str]],
    *,
    negative_type: str = "hard_visual_candidate",
) -> list[PairRecord]:
    """Label caller-supplied unrelated pairs without inventing correspondence GT."""
    by_id = {record.image_id: record for record in records}
    negatives = {
        frozenset((pair.image_a, pair.image_b)): pair
        for pair in generate_negative_pairs(by_id.values())
    }
    result = []
    for image_a, image_b in image_pairs:
        try:
            pair = negatives[frozenset((image_a, image_b))]
        except KeyError as exc:
            raise ValueError(
                f"explicit negative is missing or overlaps: {image_a}/{image_b}"
            ) from exc
        result.append(replace(pair, negative_type=negative_type))
    return result


def write_pairs(pairs: Iterable[PairRecord], path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [pair.as_dict() for pair in pairs]
    if not rows:
        raise ValueError("refusing to write an empty pair manifest")
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def read_pairs(path: str) -> list[PairRecord]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row.setdefault("label", "positive")
        row.setdefault("negative_type", None)
        row.setdefault("geographic_separation_m", None)
    return [PairRecord(**row) for row in rows]
