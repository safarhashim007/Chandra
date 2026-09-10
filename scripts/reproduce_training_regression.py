#!/usr/bin/env python3
"""Reproduce the known refiner-only loss/EPE divergence on the real NAC smoke pair."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.gt_warp import dense_pixel_warp
from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.evaluation.metrics import correspondence_metrics
from chandrappan.training.coordinates import build_gt_for_stage, roma_warp_to_pixel
from chandrappan.training.forward import forward_train
from chandrappan.training.losses import warp_huber_loss


def _crop_metadata(metadata, origin: tuple[int, int], size: int):
    world_origin = metadata.transform.pixel_to_world(*origin)
    return replace(
        metadata,
        width=size,
        height=size,
        transform=replace(
            metadata.transform,
            x_origin=float(world_origin[0]),
            y_origin=float(world_origin[1]),
        ),
    )


def _image(path: Path, origin: tuple[int, int], size: int, device: torch.device) -> torch.Tensor:
    with rasterio.open(path) as dataset:
        values = dataset.read(1, window=Window(origin[0], origin[1], size, size)).astype("float32")
    values = np.nan_to_num(values, nan=0.0)
    low, high = np.percentile(values, [1, 99])
    values = np.clip((values - low) / (high - low + 1e-6), 0.0, 1.0)
    return torch.from_numpy(np.repeat(values[None], 3, axis=0))[None].to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/training_regression.json"))
    args = parser.parse_args()
    raise RuntimeError(
        "This retired diagnostic targets the immutable T0 pair and is intentionally blocked. "
        "Use scripts/run_refiner_objective_experiment.py with a TRAIN manifest instead."
    )
    size = 320
    left = Path("data/raw/lroc_pair/NAC_PHO_E018N3346_M107042466L.TIF")
    right = Path("data/raw/lroc_pair/NAC_PHO_E018N3346_M107042466R.TIF")
    origin_left, origin_right = (5331, 17213), (155, 17303)
    metadata_left = _crop_metadata(
        read_lroc_metadata(left, left.with_suffix(".xml")), origin_left, size
    )
    metadata_right = _crop_metadata(
        read_lroc_metadata(right, right.with_suffix(".xml")), origin_right, size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_a = _image(left, origin_left, size, device)
    image_b = _image(right, origin_right, size, device)
    warp_np, valid_np = dense_pixel_warp(metadata_left, metadata_right)
    warp_px = torch.from_numpy(warp_np).to(device)[None]
    valid = torch.from_numpy(valid_np).to(device)[None]

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "romav2" / "src"))
    from romav2 import RoMaV2

    model = RoMaV2(checkpoint_path="checkpoints/romav2_official.pt")
    model.apply_setting("turbo")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for refiner in model.refiners.values():
        for parameter in refiner.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-4
    )

    def measure() -> tuple[dict[str, float | int], float]:
        output = forward_train(model, image_a, image_b)["final"]["warp_ab"]
        target, mask = build_gt_for_stage(
            warp_px, valid, (size, size), (size, size), output.shape[1:3]
        )
        pixels = roma_warp_to_pixel(output, (size, size))
        metrics = correspondence_metrics(pixels, warp_px, valid, (1, 3, 5, 10))
        return metrics, float(warp_huber_loss(output, target, mask, beta=args.beta).item())

    before, before_loss = measure()
    for _ in range(args.steps):
        output = forward_train(model, image_a, image_b)["final"]["warp_ab"]
        target, mask = build_gt_for_stage(
            warp_px, valid, (size, size), (size, size), output.shape[1:3]
        )
        loss = warp_huber_loss(output, target, mask, beta=args.beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after, after_loss = measure()
    result = {
        "steps": args.steps,
        "beta": args.beta,
        "device": str(device),
        "before_loss": before_loss,
        "after_loss": after_loss,
        "before": before,
        "after": after,
        "observations": [
            "pixel-space GT crop center is approximately [160.5, 159.5]",
            "pixel/normalized round-trip tests pass",
            "the objective is Smooth L1 in normalized coordinates; beta is recorded above",
            "official ConvRefiner detaches the previous warp between stages",
            (
                "this experiment narrows the failure to objective/optimization behavior "
                "or pair suitability; it does not prove a coordinate-direction bug"
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
