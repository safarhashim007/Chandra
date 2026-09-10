#!/usr/bin/env python3
"""Evaluate untouched RoMa v2 and the registration verifier on real validation pairs."""

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
from chandrappan.evaluation.protocol import AcceptanceConfig, accept_registration, vrr_far
from chandrappan.evaluation.t0 import sha256_file
from chandrappan.evaluation.visualize import save_match_diagnostic
from chandrappan.geo.footprint import footprint_corners_world
from chandrappan.geometry.verification import VerificationConfig, verify_registration
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


def _center(metadata_a, metadata_b) -> tuple[float, float]:
    first_geometry = Polygon(footprint_corners_world(metadata_a))
    second_geometry = Polygon(footprint_corners_world(metadata_b))
    intersection = first_geometry.intersection(second_geometry)
    if not intersection.is_empty:
        return intersection.centroid.x, intersection.centroid.y
    centroid = first_geometry.centroid
    return centroid.x, centroid.y


def _points_and_confidence(warp: torch.Tensor, confidence: torch.Tensor, crop_size: int):
    height, width = warp.shape[1:3]
    ys, xs = torch.meshgrid(
        torch.arange(height, device=warp.device, dtype=warp.dtype) + 0.5,
        torch.arange(width, device=warp.device, dtype=warp.dtype) + 0.5,
        indexing="ij",
    )
    source = torch.stack((xs * crop_size / width, ys * crop_size / height), dim=-1)[None]
    target = roma_warp_to_pixel(warp, (crop_size, crop_size))
    score = confidence.reshape(-1)
    source, target = source.reshape(-1, 2), target.reshape(-1, 2)
    finite = torch.isfinite(target).all(dim=1) & torch.isfinite(score)
    inside = (
        (target[:, 0] >= 0)
        & (target[:, 0] < crop_size)
        & (target[:, 1] >= 0)
        & (target[:, 1] < crop_size)
    )
    keep = finite & inside
    return source[keep], target[keep], score[keep]


def _inlier_mask(source: np.ndarray, target: np.ndarray, candidate) -> np.ndarray:
    if candidate is None or candidate.matrix is None:
        return np.zeros(len(source), dtype=bool)
    matrix = np.asarray(candidate.matrix, dtype=float)
    homogeneous = np.column_stack((source, np.ones(len(source))))
    projected = homogeneous @ matrix.T
    if matrix.shape == (3, 3):
        projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - target, axis=1)
    return errors <= 3.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/expanded/images.parquet"))
    parser.add_argument(
        "--positive-pairs", type=Path, default=Path("data/expanded/pairs/validation.parquet")
    )
    parser.add_argument(
        "--negative-pairs",
        type=Path,
        default=Path("data/expanded/pairs/validation_negative.parquet"),
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/romav2_official.pt"))
    parser.add_argument(
        "--refiner-state",
        type=Path,
        help="Optional refiner-only state produced by a controlled training experiment.",
    )
    parser.add_argument("--setting", default="turbo", choices=("turbo", "fast", "base"))
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--max-matches", type=int, default=4000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/validation"))
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument(
        "--acceptance-config", type=Path, default=Path("configs/registration_acceptance.yaml")
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "romav2" / "src"))
    from romav2 import RoMaV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoMaV2(checkpoint_path=str(args.checkpoint))
    model.apply_setting(args.setting)
    if args.refiner_state is not None:
        model.refiners.load_state_dict(
            torch.load(args.refiner_state, map_location=device, weights_only=True)
        )
    records = {record.image_id: record for record in read_manifest(args.manifest)}
    pairs = read_pairs(args.positive_pairs) + read_pairs(args.negative_pairs)
    if args.acceptance_config.exists():
        import yaml

        acceptance = AcceptanceConfig(
            **yaml.safe_load(args.acceptance_config.read_text())["acceptance"]
        )
    else:
        acceptance = AcceptanceConfig()
    geometry_config = VerificationConfig()
    rows = []
    positive_errors: list[np.ndarray] = []
    for pair in pairs:
        first, second = records[pair.image_a], records[pair.image_b]
        label_a = Path(first.image_path).with_suffix(".xml")
        label_b = Path(second.image_path).with_suffix(".xml")
        metadata_a = read_lroc_metadata(first.image_path, label_a if label_a.exists() else None)
        metadata_b = read_lroc_metadata(second.image_path, label_b if label_b.exists() else None)
        center = _center(metadata_a, metadata_b)
        crop_a, origin_a = _crop_metadata(metadata_a, center, args.crop_size)
        crop_b, origin_b = _crop_metadata(metadata_b, center, args.crop_size)
        image_a = _read_crop(Path(first.image_path), origin_a, args.crop_size).to(device)
        image_b = _read_crop(Path(second.image_path), origin_b, args.crop_size).to(device)
        with torch.inference_mode():
            prediction = model.match(image_a, image_b)
        source, target, confidence = _points_and_confidence(
            prediction["warp_AB"], prediction["overlap_AB"], args.crop_size
        )
        if len(source) > args.max_matches:
            selected = torch.topk(confidence, args.max_matches).indices
            source, target, confidence = source[selected], target[selected], confidence[selected]
        source_np, target_np = source.cpu().numpy(), target.cpu().numpy()
        verification = verify_registration(
            source_np,
            target_np,
            (args.crop_size, args.crop_size),
            config=geometry_config,
        )
        candidate = verification.selected
        metrics = (
            candidate.metrics()
            if candidate
            else {
                "correspondences": len(source_np),
                "inliers": 0,
                "inlier_ratio": 0.0,
                "reprojection_error_px": float("inf"),
                "spatial_coverage": 0.0,
                "grid_coverage": 0.0,
                "hull_coverage": 0.0,
                "scale": float("nan"),
                "rotation_degrees": float("nan"),
                "shear": float("nan"),
                "anisotropy": float("nan"),
            }
        )
        accepted = accept_registration(metrics, acceptance)
        row = {
            "pair_id": pair.pair_id,
            "image_a": pair.image_a,
            "image_b": pair.image_b,
            "is_positive": pair.label != "negative",
            "negative_type": pair.negative_type,
            "accepted": accepted,
            "matcher_failed": False,
            "geometric_failed": candidate is None,
            "rejection_reason": verification.rejection_reason,
            "metrics": metrics,
            "model_type": candidate.model_type if candidate else None,
            "confidence_analysis": None,
        }
        if row["is_positive"]:
            gt_np, valid_np = dense_pixel_warp(crop_a, crop_b)
            valid = torch.from_numpy(valid_np).to(device)
            predicted_px = roma_warp_to_pixel(
                prediction["warp_AB"], (args.crop_size, args.crop_size)
            )[0]
            gt = torch.from_numpy(gt_np).to(device)
            errors = torch.linalg.vector_norm(predicted_px - gt, dim=-1)[valid.bool()].cpu().numpy()
            positive_errors.append(errors)
            row["correspondence_metrics"] = correspondence_metrics(
                predicted_px[None], gt[None], valid[None], (1, 3, 5, 10)
            )
            valid_confidence = prediction["overlap_AB"][0].reshape(-1)[valid.reshape(-1)]
            row["error_distribution"] = error_distribution(errors)
            row["confidence_analysis"] = confidence_subsets(errors, valid_confidence.cpu().numpy())
        rows.append(row)
        if args.diagnostics_dir is not None and (row["is_positive"] or not accepted):
            kind = "positive" if row["is_positive"] else "negative"
            status = "accepted" if accepted else "rejected"
            save_match_diagnostic(
                image_a[0, 0].cpu().numpy(),
                image_b[0, 0].cpu().numpy(),
                source_np,
                target_np,
                _inlier_mask(source_np, target_np, candidate),
                args.diagnostics_dir / f"{status}_{kind}_{pair.pair_id}.png",
                title=f"{status} {kind}: {pair.pair_id}",
            )
        print(pair.pair_id, "positive" if row["is_positive"] else "negative", accepted)
    outcomes = [(row["is_positive"], row["accepted"]) for row in rows]
    result = {
        "dataset": "expanded_validation",
        "manifest": str(args.manifest),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": str(device),
        "setting": args.setting,
        "refiner_state": str(args.refiner_state) if args.refiner_state is not None else None,
        "acceptance_config": acceptance.as_dict(),
        "geometry_config": geometry_config.as_dict(),
        "pair_count": len(rows),
        "registration": vrr_far(outcomes),
        "correspondence_metrics": {
            "valid_correspondences": int(sum(len(errors) for errors in positive_errors)),
            "median_epe_px": float(np.median(np.concatenate(positive_errors))),
            "mean_epe_px": float(np.mean(np.concatenate(positive_errors))),
            "pck_1": float(np.mean(np.concatenate(positive_errors) <= 1.0)),
            "pck_3": float(np.mean(np.concatenate(positive_errors) <= 3.0)),
            "pck_5": float(np.mean(np.concatenate(positive_errors) <= 5.0)),
            "pck_10": float(np.mean(np.concatenate(positive_errors) <= 10.0)),
        }
        if positive_errors
        else None,
        "pairs": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    positives = [row for row in rows if row["is_positive"]]
    negatives = [row for row in rows if not row["is_positive"]]
    summary = [
        "# Real LROC validation baseline",
        "",
        f"Pairs: {len(rows)} ({len(positives)} positive, {len(negatives)} negative)",
        f"VRR: {result['registration']['vrr']}",
        f"FAR: {result['registration']['far']}",
        f"Checkpoint SHA256: `{result['checkpoint_sha256']}`",
        "",
    ]
    if positives:
        summary.extend(
            f"- {key}: {value}"
            for key, value in positives[0].get("correspondence_metrics", {}).items()
        )
        summary.extend(
            f"- {key}: {value}" for key, value in positives[0].get("error_distribution", {}).items()
        )
    (args.output_dir / "baseline.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(result["registration"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
