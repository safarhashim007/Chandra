#!/usr/bin/env python3
"""Run declared RP-001 hard-negative challenges through frozen official RoMa geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.data.manifest import read_manifest
from chandrappan.evaluation.protocol import AcceptanceConfig, accept_registration, vrr_far
from chandrappan.geometry.verification import VerificationConfig, verify_registration
from scripts.evaluate_region_pack import _crop_metadata, _overlap_center, _points, _read_crop


def _configured_test_observations(selection: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(selection.read_text(encoding="utf-8"))
    products = [row for row in config["products"] if row["role"] == "TEST"]
    result = []
    for row in products:
        img = Path(row["local_path"])
        tif = img.with_suffix(".TIF")
        result.append(
            {
                "product_id": row["product_id"],
                "raster": tif,
                "metadata": read_lroc_metadata(tif, tif.with_suffix(".xml")),
            }
        )
    return result


def _external_negatives() -> list[dict[str, Any]]:
    records = read_manifest("data/expanded/images.parquet")
    selected_regions = ("E018N3346", "E199N0308")
    result = []
    for region in selected_regions:
        record = next(row for row in records if row.region_id == region)
        tif = Path(record.image_path)
        result.append(
            {
                "product_id": record.product_id,
                "raster": tif,
                "metadata": read_lroc_metadata(tif, tif.with_suffix(".xml")),
                "negative_type": "other_lunar_region",
            }
        )
    return result


def evaluate(selection: Path, checkpoint: Path, *, crop_size: int) -> dict[str, object]:
    test = _configured_test_observations(selection)
    external = _external_negatives()
    challenges = [
        {
            "name": "other_lunar_region",
            "query": test[0],
            "candidate": external[0],
            "description": "Different mapped lunar region supplied as a retrieval candidate.",
        },
        {
            "name": "intensity_similarity_candidate",
            "query": test[1],
            "candidate": external[1],
            "description": (
                "Out-of-region lunar terrain candidate. It is a coarse photometric hard candidate, "
                "not a claim of crater-semantic similarity."
            ),
        },
        {
            "name": "same_region_nonoverlapping_crop",
            "query": test[2],
            "candidate": test[3],
            "description": "Same RP-001 maps with deliberately disjoint pixel windows.",
            "disjoint_windows": True,
        },
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "romav2" / "src"))
    from romav2 import RoMaV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoMaV2(checkpoint_path=str(checkpoint))
    model.apply_setting("precise")
    acceptance = AcceptanceConfig(
        **yaml.safe_load(Path("configs/registration_acceptance.yaml").read_text())["acceptance"]
    )
    results = []
    for challenge in challenges:
        query, candidate = challenge["query"], challenge["candidate"]
        if challenge.get("disjoint_windows"):
            query_origin = (0, 0)
            candidate_origin = (
                candidate["metadata"].width - crop_size,
                candidate["metadata"].height - crop_size,
            )
        else:
            try:
                center = _overlap_center(query["metadata"], candidate["metadata"])
            except ValueError:
                center = query["metadata"].transform.pixel_to_world(crop_size / 2, crop_size / 2)
            _, query_origin = _crop_metadata(query["metadata"], center, crop_size)
            _, candidate_origin = _crop_metadata(candidate["metadata"], center, crop_size)
        image_query, _ = _read_crop(query["raster"], query_origin, crop_size)
        image_candidate, _ = _read_crop(candidate["raster"], candidate_origin, crop_size)
        with torch.inference_mode():
            prediction = model.match(image_query.to(device), image_candidate.to(device))
        source, target, _, matching = _points(prediction, crop_size, max_matches=4000)
        verification = verify_registration(
            source, target, (crop_size, crop_size), config=VerificationConfig()
        )
        metrics = verification.selected.metrics() if verification.selected else None
        accepted = bool(
            metrics and verification.accepted and accept_registration(metrics, acceptance)
        )
        results.append(
            {
                "name": challenge["name"],
                "description": challenge["description"],
                "query_product_id": query["product_id"],
                "candidate_product_id": candidate["product_id"],
                "matching": matching,
                "verification": verification.as_dict(),
                "accepted": accepted,
            }
        )
    result = {
        "region_id": "RP-001",
        "matcher": "official RoMa v2, precise bidirectional setting",
        "available_negative_categories": [row["name"] for row in results],
        "unavailable_required_category": (
            "No validated shadow mask or independent crater-semantic similarity labels are "
            "available; "
            "those categories are not claimed as complete."
        ),
        "registration": vrr_far([(False, row["accepted"]) for row in results]),
        "cases": results,
    }
    Path("results/RP-001_hard_negative_suite.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("configs/regions/rp001.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/romav2_official.pt"))
    parser.add_argument("--crop-size", type=int, default=640)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.selection, args.checkpoint, crop_size=args.crop_size)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
