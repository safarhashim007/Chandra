#!/usr/bin/env python3
"""Run a bounded, refiner-only objective experiment on a non-T0 train pair."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.gt_warp import dense_pixel_warp
from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.data.manifest import read_manifest
from chandrappan.data.pairs import read_pairs
from chandrappan.evaluation.metrics import correspondence_metrics
from chandrappan.geo.footprint import footprint_corners_world
from chandrappan.training.coordinates import roma_warp_to_pixel
from chandrappan.training.ema import RefinerEMA
from chandrappan.training.forward import forward_train
from chandrappan.training.freezing import configure_refiner_only_training
from chandrappan.training.objective import refiner_stage_losses
from chandrappan.training.preflight import reject_t0_pair


def _crop_metadata(metadata, center: tuple[float, float], size: int):
    x, y = metadata.transform.world_to_pixel(*center)
    origin = (
        max(0, min(metadata.width - size, int(x - size / 2))),
        max(0, min(metadata.height - size, int(y - size / 2))),
    )
    world = metadata.transform.pixel_to_world(*origin)
    return replace(
        metadata,
        width=size,
        height=size,
        transform=replace(metadata.transform, x_origin=float(world[0]), y_origin=float(world[1])),
    ), origin


def _read_image(
    path: Path, origin: tuple[int, int], size: int, device: torch.device
) -> torch.Tensor:
    with rasterio.open(path) as dataset:
        image = dataset.read(1, window=Window(*origin, size, size)).astype("float32")
    image = np.nan_to_num(image, nan=0.0)
    low, high = np.percentile(image, (1, 99))
    image = np.clip((image - low) / (high - low + 1e-6), 0.0, 1.0)
    return torch.from_numpy(np.repeat(image[None], 3, axis=0))[None].to(device)


def _overlap_center(first, second) -> tuple[float, float]:
    overlap = Polygon(footprint_corners_world(first)).intersection(
        Polygon(footprint_corners_world(second))
    )
    if overlap.is_empty:
        raise ValueError("controlled training requires a geographically positive pair")
    return overlap.centroid.x, overlap.centroid.y


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(value, dict(module.named_parameters())[name].detach().cpu())
        for name, value in before.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/expanded/images.parquet"))
    parser.add_argument("--pairs", type=Path, default=Path("data/expanded/pairs/train.parquet"))
    parser.add_argument("--t0", type=Path, default=Path("benchmarks/T0_v1.parquet"))
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/romav2_official.pt"))
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("results/training_objective_fix.json"))
    parser.add_argument(
        "--state-output", type=Path, default=Path("artifacts/refiner_objective_state.pt")
    )
    args = parser.parse_args()
    if args.size != 640:
        raise ValueError(
            "this source-faithful diagnostic is intentionally fixed at RoMa base 640x640"
        )
    pairs = read_pairs(args.pairs)
    pair = pairs[args.pair_index]
    reject_t0_pair(pair.pair_id, args.t0)
    records = {record.image_id: record for record in read_manifest(args.manifest)}
    first, second = records[pair.image_a], records[pair.image_b]
    first_meta = read_lroc_metadata(first.image_path, Path(first.image_path).with_suffix(".xml"))
    second_meta = read_lroc_metadata(second.image_path, Path(second.image_path).with_suffix(".xml"))
    center = _overlap_center(first_meta, second_meta)
    first_crop, first_origin = _crop_metadata(first_meta, center, args.size)
    second_crop, second_origin = _crop_metadata(second_meta, center, args.size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_a = _read_image(Path(first.image_path), first_origin, args.size, device)
    image_b = _read_image(Path(second.image_path), second_origin, args.size, device)
    master_warp, master_valid = dense_pixel_warp(first_crop, second_crop)
    warp_px = torch.from_numpy(master_warp)[None].to(device)
    valid = torch.from_numpy(master_valid)[None].to(device)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "romav2" / "src"))
    from romav2 import RoMaV2

    model = RoMaV2(checkpoint_path=str(args.checkpoint))
    model.apply_setting("base")
    model.eval()
    configure_refiner_only_training(model)
    before_all = _snapshot(model)
    before_refiners = _snapshot(model.refiners)
    optimizer = torch.optim.AdamW(model.refiners.parameters(), lr=args.lr)
    ema = RefinerEMA(model.refiners, decay=0.999)

    def measure() -> dict[str, float | int]:
        with torch.no_grad():
            prediction = forward_train(model, image_a, image_b)["final"]["warp_ab"]
            pixels = roma_warp_to_pixel(prediction, (args.size, args.size))
            return correspondence_metrics(pixels, warp_px, valid, (1, 3, 5, 10))

    checkpoints: dict[str, object] = {"0": {"raw": measure()}}
    stopped = False
    for step in range(1, args.steps + 1):
        output = forward_train(model, image_a, image_b)
        losses = refiner_stage_losses(
            output["refiners"], warp_px, valid, (args.size, args.size), (args.size, args.size)
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        ema.update(model.refiners)
        if step in {10, 25, 50, 100, 250, args.steps}:
            raw = measure()
            with ema.average_parameters(model.refiners):
                ema_metrics = measure()
            checkpoints[str(step)] = {
                "loss": float(losses["total"].detach().item()),
                "raw": raw,
                "ema": ema_metrics,
            }
            baseline = checkpoints["0"]["raw"]
            if (
                raw["median_epe_px"] > 2.0 * baseline["median_epe_px"]
                and raw["pck_1"] < 0.5 * baseline["pck_1"]
            ):
                stopped = True
                break
    if not _changed(before_refiners, model.refiners):
        raise RuntimeError("refiner-only objective produced no refiner parameter update")
    frozen_changed = any(
        not torch.equal(value, dict(model.named_parameters())[name].detach().cpu())
        for name, value in before_all.items()
        if not name.startswith("refiners.")
    )
    if frozen_changed:
        raise RuntimeError("frozen RoMa parameters changed during refiner-only optimization")
    args.state_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.refiners.state_dict(), args.state_output)
    result = {
        "pair_id": pair.pair_id,
        "dataset": str(args.pairs),
        "t0_rejected": True,
        "device": str(device),
        "setting": "base",
        "input_size": [args.size, args.size],
        "objective": {"name": "roma_robust_warp", "alpha": 0.5, "c": 1e-3, "strides": [4, 2, 1]},
        "auxiliary_losses": {
            "overlap": "skipped: no nodata-aware overlap GT",
            "precision": "skipped: no verified covisible precision target",
        },
        "lr": args.lr,
        "ema_decay": 0.999,
        "checkpoints": checkpoints,
        "catastrophic_stop": stopped,
        "refiners_changed": True,
        "frozen_parameters_changed": False,
        "refiner_state": str(args.state_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
