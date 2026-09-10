"""Deterministic, explainable TRAIN-only lunar-pair selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from chandrappan.data.manifest import ImageRecord
from chandrappan.data.pairs import PairRecord
from chandrappan.data.scientific import SCIENTIFIC_RDR, DenseGTQuality, classify_source


@dataclass(frozen=True)
class CuratedPair:
    pair_id: str
    image_a: str
    image_b: str
    region_id: str
    selected: bool
    rejection_reason: str | None
    score: float
    score_components: dict[str, float]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def curate_train_pairs(
    records: list[ImageRecord],
    pairs: list[PairRecord],
    quality: dict[str, DenseGTQuality],
    *,
    max_pairs_per_region: int = 3,
) -> list[CuratedPair]:
    """Rank only already split TRAIN inputs; rejected rows remain visible."""
    by_id = {record.image_id: record for record in records}
    selected_per_region: dict[str, int] = {}
    ranked: list[tuple[float, PairRecord, DenseGTQuality, dict[str, float]]] = []
    rejected: list[CuratedPair] = []
    for pair in pairs:
        first, second = by_id[pair.image_a], by_id[pair.image_b]
        gt = quality[pair.pair_id]
        if (
            classify_source(first.image_path, first.product_id) != SCIENTIFIC_RDR
            or classify_source(second.image_path, second.product_id) != SCIENTIFIC_RDR
        ):
            rejected.append(
                CuratedPair(
                    pair.pair_id,
                    pair.image_a,
                    pair.image_b,
                    pair.region_id,
                    False,
                    "browse_only",
                    0,
                    {},
                    "Rejected: scientific RDR raster required.",
                )
            )
            continue
        if not gt.accepted:
            rejected.append(
                CuratedPair(
                    pair.pair_id,
                    pair.image_a,
                    pair.image_b,
                    pair.region_id,
                    False,
                    gt.rejection_reason,
                    0,
                    {},
                    "Rejected: dense geographic GT quality gate failed.",
                )
            )
            continue
        components = {
            "geometry_quality": min(1.0, pair.overlap_ratio),
            "gt_quality": gt.valid_coverage,
            "gsd_compatibility": 1.0 / pair.relative_gsd_ratio,
            "photometric_diversity": min(1.0, (pair.incidence_angle_delta or 0.0) / 30.0),
            "difficulty_score": 1.0 - min(1.0, pair.overlap_ratio),
        }
        ranked.append((sum(components.values()) / len(components), pair, gt, components))
    output = rejected[:]
    for score, pair, gt, components in sorted(ranked, key=lambda row: (-row[0], row[1].pair_id)):
        count = selected_per_region.get(pair.region_id, 0)
        if count >= max_pairs_per_region:
            output.append(
                CuratedPair(
                    pair.pair_id,
                    pair.image_a,
                    pair.image_b,
                    pair.region_id,
                    False,
                    "regional_balance",
                    score,
                    components,
                    "Eligible but not selected: region quota preserves geographic diversity.",
                )
            )
            continue
        selected_per_region[pair.region_id] = count + 1
        output.append(
            CuratedPair(
                pair.pair_id,
                pair.image_a,
                pair.image_b,
                pair.region_id,
                True,
                None,
                score,
                components,
                f"Selected: {gt.valid_coverage:.1%} valid geographic GT coverage and "
                "cycle error ≤ configured limit.",
            )
        )
    return sorted(output, key=lambda row: row.pair_id)


def freeze_selection(
    run_id: str, pairs: list[CuratedPair], destination: str | Path
) -> tuple[Path, Path]:
    """Write an immutable selected-pair manifest and its SHA256 sidecar."""
    target = Path(destination) / run_id
    target.mkdir(parents=True, exist_ok=True)
    selected = [pair.as_dict() for pair in pairs if pair.selected]
    manifest = target / "selected_training_data.json"
    manifest.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parquet = target / "selected_training_data.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(selected), parquet)
    checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    sidecar = target / "selected_training_data.sha256"
    sidecar.write_text(f"{checksum}  {manifest.name}\n", encoding="utf-8")
    return manifest, sidecar
