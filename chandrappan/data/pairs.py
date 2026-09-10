"""Geographic candidate-pair generation from canonical image manifests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
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

    return [PairRecord(**row) for row in pq.read_table(path).to_pylist()]
