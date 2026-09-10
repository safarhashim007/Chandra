#!/usr/bin/env python3
"""Evaluate untouched official RoMa v2 on the project-defined T0 pairs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from rasterio.windows import Window
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.gt_warp import dense_pixel_warp
from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.data.manifest import read_manifest
from chandrappan.data.pairs import read_pairs
from chandrappan.evaluation.analysis import confidence_subsets, error_distribution
from chandrappan.evaluation.metrics import correspondence_metrics
from chandrappan.evaluation.t0 import sha256_file, verify_checksum
from chandrappan.geo.footprint import footprint_corners_world
from chandrappan.training.coordinates import roma_warp_to_pixel


def _read_crop(path: Path, origin: tuple[int, int], size: int) -> torch.Tensor:
    import rasterio

    with rasterio.open(path) as dataset:
        values = dataset.read(1, window=Window(origin[0], origin[1], size, size)).astype("float32")
    values = np.nan_to_num(values, nan=0.0)
    low, high = np.percentile(values, [1, 99])
    values = np.clip((values - low) / (high - low + 1e-6), 0.0, 1.0)
    return torch.from_numpy(np.repeat(values[None], 3, axis=0))[None]


def _crop_metadata(metadata, center_world: tuple[float, float], size: int):
    center_px = metadata.transform.world_to_pixel(*center_world)
    origin_x = max(0, min(metadata.width - size, int(center_px[0] - size / 2)))
    origin_y = max(0, min(metadata.height - size, int(center_px[1] - size / 2)))
    world_origin = metadata.transform.pixel_to_world(origin_x, origin_y)
    return (
        replace(
            metadata,
            width=size,
            height=size,
            transform=replace(
                metadata.transform,
                x_origin=float(world_origin[0]),
                y_origin=float(world_origin[1]),
            ),
        ),
        (origin_x, origin_y),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=Path, default=Path("benchmarks/T0_v1.parquet"))
    parser.add_argument("--t0-sha256", type=Path, default=Path("benchmarks/T0_v1.sha256"))
    parser.add_argument("--manifest", type=Path, default=Path("data/generated/images.parquet"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/romav2_official.pt"))
    parser.add_argument("--setting", default="turbo", choices=("turbo", "fast", "base"))
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--output-dir", type=Path, default=Path("results/T0"))
    args = parser.parse_args()
    verify_checksum(args.t0, args.t0_sha256)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "romav2" / "src"))
    from romav2 import RoMaV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoMaV2(checkpoint_path=str(args.checkpoint))
    model.apply_setting(args.setting)
    records = {record.image_id: record for record in read_manifest(args.manifest)}
    pair_rows = read_pairs(args.t0)
    pair_results = []
    for pair in pair_rows:
        first, second = records[pair.image_a], records[pair.image_b]
        metadata_a = read_lroc_metadata(
            first.image_path, Path(first.image_path).with_suffix(".xml")
        )
        metadata_b = read_lroc_metadata(
            second.image_path, Path(second.image_path).with_suffix(".xml")
        )
        overlap = Polygon(footprint_corners_world(metadata_a)).intersection(
            Polygon(footprint_corners_world(metadata_b))
        )
        center = overlap.centroid.x, overlap.centroid.y
        crop_a, origin_a = _crop_metadata(metadata_a, center, args.crop_size)
        crop_b, origin_b = _crop_metadata(metadata_b, center, args.crop_size)
        image_a = _read_crop(Path(first.image_path), origin_a, args.crop_size).to(device)
        image_b = _read_crop(Path(second.image_path), origin_b, args.crop_size).to(device)
        with torch.inference_mode():
            prediction = model.match(image_a, image_b)
        gt_np, valid_np = dense_pixel_warp(crop_a, crop_b)
        gt = torch.from_numpy(gt_np).to(device)[None]
        valid = torch.from_numpy(valid_np).to(device)[None]
        predicted = roma_warp_to_pixel(prediction["warp_AB"], (args.crop_size, args.crop_size))
        metrics = correspondence_metrics(predicted, gt, valid, (1, 3, 5, 10))
        errors = torch.linalg.vector_norm(predicted - gt, dim=-1)[valid.bool()].cpu().numpy()
        confidence = prediction["overlap_AB"][0].reshape(-1)[valid.reshape(-1)].cpu().numpy()
        pair_results.append(
            {
                "pair_id": pair.pair_id,
                "metrics": metrics,
                "valid_registration": None,
                "geometric_metrics": None,
                "error_distribution": error_distribution(errors),
                "confidence_analysis": confidence_subsets(errors, confidence),
                "note": (
                    "T0 remains positive-only; acceptance thresholds are selected on "
                    "validation, never T0."
                ),
            }
        )

    result = {
        "benchmark": "T0_v1",
        "benchmark_sha256": sha256_file(args.t0),
        "benchmark_kind": "project-defined smoke benchmark; not official LunarMatch-NASA",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": str(device),
        "setting": args.setting,
        "pair_count": len(pair_results),
        "successfully_evaluated_pairs": len(pair_results),
        "failed_pairs": 0,
        "pairs": pair_results,
        "vrr": None,
        "far": None,
        "geometric_inliers": None,
        "inlier_ratio": None,
        "geometric_reprojection_error_px": None,
        "spatial_coverage": None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# T0 official RoMa v2 baseline",
        "",
        f"Benchmark SHA256: `{result['benchmark_sha256']}`",
        f"Checkpoint SHA256: `{result['checkpoint_sha256']}`",
        (
            f"Pairs: {result['pair_count']}; successfully evaluated: "
            f"{result['successfully_evaluated_pairs']}"
        ),
        "",
        (
            "This is a project-defined smoke benchmark over the currently available real "
            "LROC pair, not the official LunarMatch-NASA benchmark."
        ),
        (
            "T0 is reported without threshold tuning; registration thresholds are selected "
            "on the separate validation corpus."
        ),
        "",
    ]
    for row in pair_results:
        lines.append(f"## {row['pair_id']}")
        for key, value in row["metrics"].items():
            lines.append(f"- {key}: {value}")
    (args.output_dir / "baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
