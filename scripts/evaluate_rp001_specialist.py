#!/usr/bin/env python3
"""Paired official-vs-D1 evaluation for the frozen RP-001 specialist."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.geometry.verification import VerificationConfig, verify_registration
from chandrappan.rp001_demo import as_roma_tensor, canonical_json_sha256, preprocess_science
from chandrappan.training.coordinates import roma_warp_to_pixel
from scripts.run_rp001_demo_adaptation import (
    _arrays,
    _evaluation_samples,
    _load_manifest,
    _load_model,
    _monitor_crops,
)


def _aggregate(values: list[np.ndarray]) -> dict[str, float | int]:
    errors = np.concatenate(values)
    return {
        "valid_correspondences": int(len(errors)),
        "median_epe_px": float(np.median(errors)),
        "mean_epe_px": float(np.mean(errors)),
        "pck_0.5": float(np.mean(errors <= 0.5)),
        "pck_1": float(np.mean(errors <= 1.0)),
        "pck_3": float(np.mean(errors <= 3.0)),
    }


def _evaluate(
    model: Any, samples: list[dict[str, Any]], preprocess: str, device: torch.device
) -> dict[str, Any]:
    all_errors, rows = [], []
    model.eval()
    for sample in samples:
        source, target, ground_truth, valid = _arrays(sample)
        with torch.inference_mode():
            prediction = model.match(
                as_roma_tensor(preprocess_science(source, preprocess), device),
                as_roma_tensor(preprocess_science(target, preprocess), device),
            )
        warp = roma_warp_to_pixel(prediction["warp_AB"], (640, 640))[0].detach().cpu().numpy()
        errors = np.linalg.norm(warp - ground_truth, axis=-1)[valid]
        if not np.isfinite(errors).all():
            raise FloatingPointError(f"non-finite error for {sample['sample_id']}")
        all_errors.append(errors)
        confidence = prediction["overlap_AB"][0, ..., 0].detach().cpu().numpy()
        yy, xx = np.indices(warp.shape[:2], dtype=np.float32)
        source_xy = np.stack((xx + 0.5, yy + 0.5), axis=-1)
        inside = (
            (warp[..., 0] >= 0) & (warp[..., 0] < 640) & (warp[..., 1] >= 0) & (warp[..., 1] < 640)
        )
        keep = (
            valid
            & inside
            & np.isfinite(warp).all(axis=-1)
            & np.isfinite(confidence)
            & (confidence >= 0.5)
        )
        flat = np.flatnonzero(keep)
        if len(flat) > 4000:
            flat = flat[np.argsort(-confidence.reshape(-1)[flat], kind="stable")[:4000]]
        verification = verify_registration(
            source_xy.reshape(-1, 2)[flat],
            warp.reshape(-1, 2)[flat],
            (640, 640),
            config=VerificationConfig(),
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "valid_gt_coverage": float(valid.mean()),
                "metric": {
                    "median_epe_px": float(np.median(errors)),
                    "pck_0.5": float(np.mean(errors <= 0.5)),
                    "pck_1": float(np.mean(errors <= 1.0)),
                },
                "geometry_accepted": verification.accepted,
                "verification": verification.as_dict(),
            }
        )
    accepted = sum(row["geometry_accepted"] for row in rows)
    return {
        "dense_gt": _aggregate(all_errors),
        "positive_vrr": {
            "accepted": accepted,
            "positive_queries": len(rows),
            "vrr": accepted / len(rows) if rows else 0.0,
            "protocol": (
                "base-640 one-directional RoMa dense matches, confidence >= 0.5, "
                "project geometric verification"
            ),
        },
        "queries": rows,
    }


def evaluate(selection: Path, run_dir: Path, checkpoint: Path) -> dict[str, Any]:
    result = json.loads((run_dir / "adaptation_result.json").read_text(encoding="utf-8"))
    manifest, crops = _load_manifest(run_dir)
    samples = {
        "TRAIN": _monitor_crops(crops),
        "VALIDATION": _evaluation_samples(selection, "VALIDATION")[0],
        "TEST": _evaluation_samples(selection, "TEST")[0],
    }
    model, device = _load_model(checkpoint)
    official_stride1 = copy.deepcopy(model.refiners["1"].state_dict())
    specialist = torch.load(
        run_dir / "rp001_specialist_stride1.pt", map_location=device, weights_only=True
    )
    comparison = {}
    for split, split_samples in samples.items():
        model.refiners["1"].load_state_dict(official_stride1)
        official = _evaluate(model, split_samples, result["preprocessing"]["selected"], device)
        model.refiners["1"].load_state_dict(specialist)
        regional = _evaluate(model, split_samples, result["preprocessing"]["selected"], device)
        comparison[split] = {"official_romav2": official, "rp001_d1": regional}
    payload = {
        "region_id": "RP-001",
        "scope": "experimental regional specialist; not production or global evidence",
        "selected_checkpoint": result["selected_checkpoint"],
        "preprocessing": result["preprocessing"]["selected"],
        "manifest_sha256": manifest["manifest_sha256"],
        "comparison": comparison,
    }
    payload["sha256"] = canonical_json_sha256(payload)
    (run_dir / "paired_evaluation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("configs/regions/rp001.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/rp001_demo_adaptation_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/romav2_official.pt"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.selection, args.run_dir, args.checkpoint), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
