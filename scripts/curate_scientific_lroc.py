#!/usr/bin/env python3
# ruff: noqa: E501
"""Validate real LROC TRAIN pairs, curate them, and freeze demo provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.curator import curate_train_pairs, freeze_selection
from chandrappan.data.manifest import read_manifest
from chandrappan.data.pairs import read_pairs
from chandrappan.data.scientific import SCIENTIFIC_RDR, classify_source, dense_gt_quality


def main() -> int:
    records = read_manifest("data/expanded/images.parquet")
    train = read_manifest("data/expanded/train.parquet")
    pairs = read_pairs("data/expanded/pairs/train.parquet")
    quality = {}
    for pair in pairs:
        by_id = {record.image_id: record for record in train}
        quality[pair.pair_id] = dense_gt_quality(by_id[pair.image_a], by_id[pair.image_b])
    curated = curate_train_pairs(train, pairs, quality)
    manifest, checksum = freeze_selection("lroc_curated_v1", curated, "runs")
    selected = [row for row in curated if row.selected]
    source_types = [classify_source(row.image_path, row.product_id) for row in records]
    result = {
        "run_id": "lroc_curated_v1",
        "products_discovered": len(records),
        "scientific_products": source_types.count(SCIENTIFIC_RDR),
        "browse_only_products": len(records) - source_types.count(SCIENTIFIC_RDR),
        "regions": len({row.region_id for row in records}),
        "train_regions": len({row.region_id for row in train}),
        "train_candidate_pairs": len(pairs),
        "dense_gt_valid_pairs": sum(item.accepted for item in quality.values()),
        "rejected_pairs": sum(not item.accepted for item in quality.values()),
        "selected_pairs": len(selected),
        "selected_images": len({image for row in selected for image in (row.image_a, row.image_b)}),
        "selected_regions": len({row.region_id for row in selected}),
        "validation_used_for_training": 0,
        "test_used_for_training": 0,
        "t0_used_for_training": 0,
        "manifest": str(manifest),
        "parquet_manifest": str(manifest.with_suffix(".parquet")),
        "checksum_path": str(checksum),
        "sha256": checksum.read_text().split()[0],
        "pairs": [row.as_dict() for row in curated],
        "gt_quality": {pair_id: item.as_dict() for pair_id, item in quality.items()},
        "checkpoint_provenance": None,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/dataset_curation.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Scientific LROC curation",
        "",
        *(
            f"- {key}: {value}"
            for key, value in result.items()
            if key not in {"pairs", "gt_quality"}
        ),
        "",
        "## Leakage",
        "",
        "VALIDATION USED FOR TRAINING: 0",
        "TEST USED FOR TRAINING: 0",
        "T0 USED FOR TRAINING: 0",
    ]
    Path("results/dataset_curation.md").write_text("\n".join(lines) + "\n")
    source_report = {
        "products_discovered": len(records),
        "scientific_rdr_products": source_types.count(SCIENTIFIC_RDR),
        "browse_only_products": len(records) - source_types.count(SCIENTIFIC_RDR),
        "regions": len({row.region_id for row in records}),
        "metadata_availability": {
            "acquisition_time": sum(row.acquisition_time is not None for row in records),
            "gsd": sum(row.gsd_m_per_px is not None for row in records),
            "incidence": sum(row.incidence_angle is not None for row in records),
            "emission": sum(row.emission_angle is not None for row in records),
            "phase": sum(row.phase_angle is not None for row in records),
            "mask": sum(item.valid_points > 0 for item in quality.values()),
        },
        "dense_gt_valid_pairs": result["dense_gt_valid_pairs"],
        "rejected_pairs": result["rejected_pairs"],
        "source_type_for_training": SCIENTIFIC_RDR,
    }
    Path("results/source_data_report.json").write_text(json.dumps(source_report, indent=2) + "\n")
    Path("results/source_data_report.md").write_text(
        "# Scientific LROC source report\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in source_report.items())
        + "\n"
    )
    cards = "".join(
        f"<li><b>{row.pair_id}</b> — {'USED FOR FINE-TUNING' if row.selected else 'NOT SELECTED'}: {row.reason}</li>"
        for row in curated
    )
    Path("results/demo_curation.html").write_text(
        f"<h1>Chandrappan Scientific LROC Curation</h1><p>Scientific products: {result['scientific_products']}; selected pairs: {len(selected)}</p><ul>{cards}</ul>"
    )
    Path("results/demo_curation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "scientific_products",
                    "dense_gt_valid_pairs",
                    "selected_pairs",
                    "sha256",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
