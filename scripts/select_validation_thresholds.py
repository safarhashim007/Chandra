#!/usr/bin/env python3
"""Select registration thresholds from validation rows only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.evaluation.protocol import AcceptanceConfig
from chandrappan.evaluation.thresholds import select_validation_thresholds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/validation/baseline.json"))
    parser.add_argument("--dataset-name", default="expanded_validation")
    parser.add_argument("--far-limit", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("configs/registration_acceptance.yaml"))
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = [
        {"is_positive": row["is_positive"], "metrics": row["metrics"]} for row in payload["pairs"]
    ]
    candidates = []
    for minimums in product((100, 500, 1000), (30, 100, 500), (0.25, 0.5), (0.10, 0.20)):
        candidates.append(
            AcceptanceConfig(
                min_correspondences=minimums[0],
                min_inliers=minimums[1],
                min_inlier_ratio=minimums[2],
                min_spatial_coverage=minimums[3],
                min_grid_coverage=minimums[3],
            )
        )
    selected = select_validation_thresholds(
        rows,
        candidates,
        far_limit=args.far_limit,
        dataset_name=args.dataset_name,
    )
    output = {
        "dataset": args.dataset_name,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "far_limit": args.far_limit,
        "search_candidate_count": len(candidates),
        "achieved_vrr": selected.vrr,
        "achieved_far": selected.far,
        "positive_count": selected.positive_count,
        "negative_count": selected.negative_count,
        "acceptance": selected.config.as_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(output, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
