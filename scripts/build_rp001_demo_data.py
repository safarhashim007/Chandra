#!/usr/bin/env python3
"""Freeze all usable RP-001 TRAIN pairs and geometry-preserving crop samples.

This is an experimental, region-specific data builder.  It intentionally keeps
every validated TRAIN pair rather than applying the multi-region curator's
diversity quota.  Validation and test acquisitions are never written to its
training crop manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.gt_warp import dense_pixel_warp
from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.evaluation.metrics import correspondence_metrics
from chandrappan.geo.footprint import footprint_corners_world
from chandrappan.rp001_demo import (
    as_roma_tensor,
    canonical_json_sha256,
    preprocess_science,
    sha256_file,
)
from chandrappan.training.coordinates import roma_warp_to_pixel


def crop_metadata(
    metadata: Any, center_world: tuple[float, float], size: int
) -> tuple[Any, tuple[int, int]]:
    """Return an exact crop-local map transform and its source-pixel origin."""
    center_x, center_y = metadata.transform.world_to_pixel(*center_world)
    origin_x = max(0, min(metadata.width - size, int(center_x - size / 2)))
    origin_y = max(0, min(metadata.height - size, int(center_y - size / 2)))
    world_origin = metadata.transform.pixel_to_world(origin_x, origin_y)
    return (
        replace(
            metadata,
            width=size,
            height=size,
            transform=replace(
                metadata.transform, x_origin=float(world_origin[0]), y_origin=float(world_origin[1])
            ),
        ),
        (origin_x, origin_y),
    )


def _load_observations(selection: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(selection.read_text(encoding="utf-8"))
    if config.get("region_id") != "RP-001":
        raise ValueError("RP-001 demo builder only accepts the RP-001 selection")
    root = Path(config["products"][0]["local_path"]).parents[1]
    with (root / "observation_catalog.csv").open(newline="", encoding="utf-8") as handle:
        catalog = {row["product_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for product in config["products"]:
        row = catalog.get(product["product_id"])
        if row is None:
            raise ValueError(f"missing built observation catalog entry: {product['product_id']}")
        metadata_path = root / "metadata" / f"{product['product_id']}.json"
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        raster = Path(metadata_payload["processed_science_raster"])
        if not raster.exists():
            raise FileNotFoundError(f"missing processed science raster: {raster}")
        rows.append(
            {
                "product_id": product["product_id"],
                "role": product["role"],
                "acquisition_date": product["acquisition_date"],
                "incidence_deg": float(product["incidence_deg"]),
                "emission_deg": float(product["emission_deg"]),
                "phase_deg": float(product["phase_deg"]),
                "raster": raster,
                "raster_sha256": sha256_file(raster),
                "metadata": read_lroc_metadata(
                    raster, Path(metadata_payload["raw_geotiff"]).with_suffix(".xml")
                ),
            }
        )
    return rows


def _read_window(raster: Path, origin: tuple[int, int], size: int) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(raster) as dataset:
        window = Window(origin[0], origin[1], size, size)
        image = dataset.read(1, window=window).astype(np.float32)
        valid = dataset.read_masks(1, window=window) > 0
    return image, valid & np.isfinite(image)


def _cycle_error(source: Any, target: Any, warp: np.ndarray, valid: np.ndarray) -> float:
    tx, ty = warp[..., 0], warp[..., 1]
    world_x, world_y = target.transform.pixel_to_world(tx, ty)
    sx, sy = source.transform.world_to_pixel(world_x, world_y)
    y, x = np.indices(valid.shape, dtype=np.float64)
    errors = np.hypot(sx - (x + 0.5), sy - (y + 0.5))[valid]
    if not errors.size:
        raise ValueError("crop has no cycle-valid GT pixels")
    return float(np.max(errors))


def _target_mask_at_warp(target_mask: np.ndarray, warp: np.ndarray) -> np.ndarray:
    x = np.rint(warp[..., 0] - 0.5).astype(int)
    y = np.rint(warp[..., 1] - 0.5).astype(int)
    inside = (x >= 0) & (x < target_mask.shape[1]) & (y >= 0) & (y < target_mask.shape[0])
    values = np.zeros_like(inside, dtype=bool)
    values[inside] = target_mask[y[inside], x[inside]]
    return values


def _mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        return dataset.read_masks(1) > 0


def _window_mean(mask: np.ndarray, size: int) -> np.ndarray:
    """Return every valid-mask mean for a fixed-size window in O(HW)."""
    integral = np.pad(mask.astype(np.uint32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return sums / float(size * size)


def _crop_candidates(
    source: dict[str, Any], target: dict[str, Any], size: int, *, minimum_coverage: float = 0.95
) -> list[tuple[str, tuple[float, float]]]:
    """Pick distinct, mask-valid 640px windows from common mapped ground.

    RP-001 products share a map grid, but their scientific pixel masks differ
    sharply.  Footprint overlap alone is therefore insufficient: candidates
    are selected from the intersection of their real raster masks.
    """
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum crop coverage must be in (0, 1]")
    source_footprint = Polygon(footprint_corners_world(source["metadata"]))
    target_footprint = Polygon(footprint_corners_world(target["metadata"]))
    overlap = source_footprint.intersection(target_footprint)
    if overlap.is_empty or overlap.area <= 0:
        return []
    source_mask, target_mask = _mask(source["raster"]), _mask(target["raster"])
    if source_mask.shape != target_mask.shape:
        raise ValueError("RP-001 mapped rasters do not share a pixel grid")
    coverage = _window_mean(source_mask & target_mask, size)
    candidates = []
    for y in range(0, coverage.shape[0], 80):
        for x in range(0, coverage.shape[1], 80):
            if coverage[y, x] < minimum_coverage:
                continue
            center = source["metadata"].transform.pixel_to_world(x + size / 2, y + size / 2)
            if overlap.covers(Point(center)):
                candidates.append((float(coverage[y, x]), x, y, center))
    if not candidates:
        return []

    chosen: list[tuple[str, tuple[float, float]]] = []
    chosen_origins: list[tuple[int, int]] = []
    targets = [
        ("central", 0.50, 0.50),
        ("upper_left", 0.22, 0.22),
        ("upper_right", 0.78, 0.22),
        ("lower_left", 0.22, 0.78),
        ("lower_right", 0.78, 0.78),
    ]
    for label, desired_x, desired_y in targets:
        ranked = sorted(
            candidates,
            key=lambda item: (
                -item[0]
                + 0.35
                * np.hypot(
                    (item[1] + size / 2) / source_mask.shape[1] - desired_x,
                    (item[2] + size / 2) / source_mask.shape[0] - desired_y,
                ),
                item[2],
                item[1],
            ),
        )
        selected = next(
            (
                item
                for item in ranked
                if all(np.hypot(item[1] - x, item[2] - y) >= size / 2 for x, y in chosen_origins)
            ),
            None,
        )
        if selected is not None:
            _, x, y, center = selected
            chosen_origins.append((x, y))
            chosen.append((label, (float(center[0]), float(center[1]))))
    texture_options = []
    for _, x, y, center in candidates[:: max(1, len(candidates) // 40)]:
        if any(
            np.hypot(x - prior_x, y - prior_y) < size / 2 for prior_x, prior_y in chosen_origins
        ):
            continue
        image, valid = _read_window(source["raster"], (x, y), size)
        values = image[valid]
        texture_options.append((float(np.std(values)) if values.size else -np.inf, x, y, center))
    if texture_options:
        _, x, y, center = max(texture_options, key=lambda item: (item[0], -item[2], -item[1]))
        chosen.append(("high_texture", (float(center[0]), float(center[1]))))
    return chosen


def _raster_statistics(path: Path) -> dict[str, float]:
    with rasterio.open(path) as dataset:
        image = dataset.read(1, out_shape=(1, 512, 512)).astype(np.float32)
        mask = dataset.read_masks(1, out_shape=(1, 512, 512)) > 0
    values = image[mask & np.isfinite(image)]
    if not values.size:
        raise ValueError(f"science raster has no finite valid values: {path}")
    return {
        "brightness_median": float(np.median(values)),
        "contrast_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    }


def _baseline_metrics(
    samples: list[dict[str, Any]], checkpoint: Path
) -> dict[str, dict[str, float | int]]:
    """Measure official RoMa v2 on each centre crop for difficulty weighting."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "romav2" / "src"))
    from romav2 import RoMaV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoMaV2(checkpoint_path=str(checkpoint))
    model.apply_setting("base")
    result: dict[str, dict[str, float | int]] = {}
    for index, sample in enumerate(samples, start=1):
        first, _ = _read_window(Path(sample["source_raster"]), tuple(sample["source_origin"]), 640)
        second, _ = _read_window(Path(sample["target_raster"]), tuple(sample["target_origin"]), 640)
        with np.load(sample["gt_path"]) as ground_truth:
            warp = torch.from_numpy(ground_truth["warp_ab_px"])[None].to(device)
            valid = torch.from_numpy(ground_truth["valid_ab"])[None].to(device)
        with torch.inference_mode():
            prediction = model.match(
                as_roma_tensor(preprocess_science(first, "raw"), device),
                as_roma_tensor(preprocess_science(second, "raw"), device),
            )
        pixels = roma_warp_to_pixel(prediction["warp_AB"], (640, 640))
        result[sample["pair_id"]] = correspondence_metrics(pixels, warp, valid, (0.5, 1.0, 3.0))
        print(f"official baseline {index}/{len(samples)}: {sample['pair_id']}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    flattened = []
    for row in rows:
        flattened.append(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    pd.DataFrame(flattened).to_parquet(path, index=False)


def build(
    selection: Path, output: Path, checkpoint: Path, *, crop_size: int = 640
) -> dict[str, Any]:
    if crop_size != 640:
        raise ValueError("RP-001 D1 training is fixed at source-faithful 640x640 inputs")
    observations = _load_observations(selection)
    train = [row for row in observations if row["role"] == "TRAIN"]
    if len(train) != 12:
        raise ValueError(f"expected exactly 12 frozen TRAIN acquisitions, found {len(train)}")
    if any(row["role"] != "TRAIN" for row in train):
        raise AssertionError("training manifest includes non-TRAIN acquisition")
    output.mkdir(parents=True, exist_ok=True)
    gt_root = output / "crop_gt"
    gt_root.mkdir(parents=True, exist_ok=True)
    observation_stats = {row["product_id"]: _raster_statistics(row["raster"]) for row in train}
    pairs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    crops: list[dict[str, Any]] = []
    for first, second in combinations(train, 2):
        pair_id = f"{first['product_id']}__{second['product_id']}"
        try:
            source_area = Polygon(footprint_corners_world(first["metadata"])).area
            target_area = Polygon(footprint_corners_world(second["metadata"])).area
            overlap = Polygon(footprint_corners_world(first["metadata"])).intersection(
                Polygon(footprint_corners_world(second["metadata"]))
            )
            overlap_fraction = float(overlap.area / min(source_area, target_area))
            if overlap_fraction <= 0:
                raise ValueError("no_actual_geographic_overlap")
            pair_crops = []
            seen_origins: set[tuple[int, int, int, int]] = set()
            for location, center in _crop_candidates(first, second, crop_size):
                source_meta, source_origin = crop_metadata(first["metadata"], center, crop_size)
                target_meta, target_origin = crop_metadata(second["metadata"], center, crop_size)
                origin_key = (*source_origin, *target_origin)
                if origin_key in seen_origins:
                    continue
                seen_origins.add(origin_key)
                source_image, source_valid = _read_window(first["raster"], source_origin, crop_size)
                target_image, target_valid = _read_window(
                    second["raster"], target_origin, crop_size
                )
                warp, geo_valid = dense_pixel_warp(source_meta, target_meta)
                valid = geo_valid & source_valid & _target_mask_at_warp(target_valid, warp)
                coverage = float(valid.mean())
                if coverage < 0.95:
                    raise ValueError(f"insufficient_nodata_aware_gt_coverage:{coverage:.6f}")
                cycle_max = _cycle_error(source_meta, target_meta, warp, valid)
                if cycle_max >= 0.01:
                    raise ValueError(f"gt_cycle_failure:{cycle_max:.6g}")
                gt_path = gt_root / f"{pair_id}__{location}.npz"
                np.savez_compressed(gt_path, warp_ab_px=warp, valid_ab=valid)
                pair_crops.append(
                    {
                        "sample_id": f"{pair_id}__{location}",
                        "pair_id": pair_id,
                        "location": location,
                        "source_product_id": first["product_id"],
                        "target_product_id": second["product_id"],
                        "source_role": "TRAIN",
                        "target_role": "TRAIN",
                        "source_raster": str(first["raster"]),
                        "target_raster": str(second["raster"]),
                        "source_origin": list(source_origin),
                        "target_origin": list(target_origin),
                        "center_world_m": [center[0], center[1]],
                        "crop_size_px": crop_size,
                        "gt_path": str(gt_path),
                        "gt_valid_coverage": coverage,
                        "gt_cycle_max_px": cycle_max,
                    }
                )
            if not pair_crops:
                raise ValueError("no_nodata_aware_640_window_at_minimum_coverage")
            centre = next(row for row in pair_crops if row["location"] == "central")
            pairs.append(
                {
                    "pair_id": pair_id,
                    "source_product_id": first["product_id"],
                    "target_product_id": second["product_id"],
                    "source_acquisition_date": first["acquisition_date"],
                    "target_acquisition_date": second["acquisition_date"],
                    "scientific_raster_validation": "PASS",
                    "actual_geographic_overlap_fraction": overlap_fraction,
                    "gt_valid": True,
                    "central_gt_valid_coverage": centre["gt_valid_coverage"],
                    "central_gt_cycle_max_px": centre["gt_cycle_max_px"],
                    "crop_count": len(pair_crops),
                    "photometry": {
                        "source": observation_stats[first["product_id"]],
                        "target": observation_stats[second["product_id"]],
                        "incidence_difference_deg": abs(
                            first["incidence_deg"] - second["incidence_deg"]
                        ),
                        "emission_difference_deg": abs(
                            first["emission_deg"] - second["emission_deg"]
                        ),
                        "phase_difference_deg": abs(first["phase_deg"] - second["phase_deg"]),
                    },
                }
            )
            crops.extend(pair_crops)
        except (OSError, ValueError) as exc:
            rejected.append(
                {
                    "pair_id": pair_id,
                    "source_product_id": first["product_id"],
                    "target_product_id": second["product_id"],
                    "reason": str(exc),
                }
            )
    if not pairs:
        raise RuntimeError("no RP-001 TRAIN pair passed scientific/GT validation")
    centre_samples = [row for row in crops if row["location"] == "central"]
    baseline = _baseline_metrics(centre_samples, checkpoint)
    baseline_pck = np.asarray([float(baseline[row["pair_id"]]["pck_1"]) for row in pairs])
    low, high = np.quantile(baseline_pck, (1 / 3, 2 / 3))
    brightness_scale = max(
        1e-6,
        max(values["brightness_median"] for values in observation_stats.values())
        - min(values["brightness_median"] for values in observation_stats.values()),
    )
    contrast_scale = max(
        1e-6,
        max(values["contrast_iqr"] for values in observation_stats.values())
        - min(values["contrast_iqr"] for values in observation_stats.values()),
    )
    raw_weights = []
    for pair in pairs:
        metrics = baseline[pair["pair_id"]]
        pck1 = float(metrics["pck_1"])
        category = "EASY" if pck1 >= high else "HARD_VALID" if pck1 <= low else "MEDIUM"
        photo = pair["photometry"]
        angular = min(
            1.0,
            (
                photo["incidence_difference_deg"]
                + photo["emission_difference_deg"]
                + photo["phase_difference_deg"]
            )
            / 120.0,
        )
        brightness = min(
            1.0,
            abs(photo["source"]["brightness_median"] - photo["target"]["brightness_median"])
            / brightness_scale,
        )
        contrast = min(
            1.0,
            abs(photo["source"]["contrast_iqr"] - photo["target"]["contrast_iqr"]) / contrast_scale,
        )
        gt_quality = float(pair["central_gt_valid_coverage"]) * (
            1.0 / (1.0 + 100.0 * float(pair["central_gt_cycle_max_px"]))
        )
        difficulty = {"EASY": 0.90, "MEDIUM": 1.00, "HARD_VALID": 1.05}[category]
        raw_weight = (
            0.60 + 0.40 * gt_quality + 0.45 * angular + 0.20 * brightness + 0.20 * contrast
        ) * difficulty
        pair["official_romav2_baseline"] = metrics
        pair["difficulty_category"] = category
        pair["weight_components"] = {
            "gt_quality": gt_quality,
            "angular_diversity": angular,
            "brightness_diversity": brightness,
            "contrast_diversity": contrast,
            "difficulty_multiplier": difficulty,
        }
        raw_weights.append(raw_weight)
    normalizer = float(np.mean(raw_weights))
    pair_weights = {pair["pair_id"]: raw / normalizer for pair, raw in zip(pairs, raw_weights)}
    for pair in pairs:
        pair["training_weight"] = pair_weights[pair["pair_id"]]
    for crop in crops:
        crop["pair_training_weight"] = pair_weights[crop["pair_id"]]
    manifest = {
        "schema_version": 1,
        "status": "EXPERIMENTAL_REGION_SPECIFIC_ADAPTATION_INPUT",
        "region_id": "RP-001",
        "site_name": "Apollo 15 S-IVB Impact",
        "split_policy": "whole acquisition; TRAIN only for this manifest",
        "train_acquisition_count": len(train),
        "theoretical_pair_count": len(train) * (len(train) - 1) // 2,
        "gt_valid_pair_count": len(pairs),
        "rejected_pair_count": len(rejected),
        "rejected_pairs": rejected,
        "valid_pairs": pairs,
        "crop_sample_count": len(crops),
        "crop_manifest": "training_crops.json",
        "official_baseline": {
            "checkpoint_sha256": sha256_file(checkpoint),
            "setting": "base_640_unidirectional",
            "purpose": "difficulty weighting only; no pair is discarded for score",
        },
        "weighting": {
            "sampling": "weighted random by valid pair; all valid pairs retained",
            "categories": ["EASY", "MEDIUM", "HARD_VALID"],
            "mean_normalized_weight": float(np.mean(list(pair_weights.values()))),
        },
    }
    crop_manifest = {
        "schema_version": 1,
        "region_id": "RP-001",
        "role_constraint": "TRAIN only",
        "samples": crops,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (output / "all_valid_training_pairs.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "training_crops.json").write_text(
        json.dumps(crop_manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_parquet(pairs, output / "all_valid_training_pairs.parquet")
    _write_parquet(crops, output / "training_crops.parquet")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("configs/regions/rp001.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/rp001_demo_adaptation_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/romav2_official.pt"))
    args = parser.parse_args()
    result = build(args.selection, args.output, args.checkpoint)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
