#!/usr/bin/env python3
"""Train and select the experimental, stride-1-only RP-001 specialist.

The script deliberately separates four stages: validation-only preprocessing
selection, TRAIN-only optimization, validation-only checkpoint selection, and
one final held-out TEST evaluation.  It never modifies the official RoMa v2
weights or inference implementation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.gt_warp import dense_pixel_warp
from chandrappan.region_packs import intensity_descriptor, rank_references
from chandrappan.rp001_demo import (
    PREPROCESSING_METHODS,
    as_roma_tensor,
    canonical_json_sha256,
    photometric_augment,
    preprocess_science,
    sha256_file,
)
from chandrappan.training.coordinates import roma_warp_to_pixel
from chandrappan.training.ema import RefinerEMA
from chandrappan.training.forward import forward_train
from chandrappan.training.freezing import configure_stride1_refiner_training
from chandrappan.training.objective import refiner_stage_losses
from scripts.build_rp001_demo_data import (
    _crop_candidates,
    _load_observations,
    _read_window,
    _target_mask_at_warp,
    crop_metadata,
)


def _load_manifest(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((run_dir / "all_valid_training_pairs.json").read_text(encoding="utf-8"))
    crops = json.loads((run_dir / "training_crops.json").read_text(encoding="utf-8"))["samples"]
    if manifest["status"] != "EXPERIMENTAL_REGION_SPECIFIC_ADAPTATION_INPUT":
        raise ValueError("unexpected RP-001 adaptation manifest status")
    if manifest["train_acquisition_count"] != 12:
        raise ValueError("RP-001 adaptation must use exactly the frozen 12 TRAIN acquisitions")
    if any(row["source_role"] != "TRAIN" or row["target_role"] != "TRAIN" for row in crops):
        raise ValueError("TRAIN manifest contains validation or test acquisition")
    if not crops:
        raise ValueError("RP-001 adaptation has no geometrically valid crop samples")
    return manifest, crops


def _descriptor(path: Path) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        image = dataset.read(1, out_shape=(1, 256, 256), resampling=Resampling.average)
    return intensity_descriptor(image)


def _evaluation_samples(
    selection: Path, role: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make auto-retrieved, GT-valid central samples for VALIDATION or TEST only."""
    if role not in {"VALIDATION", "TEST"}:
        raise ValueError("evaluation role must be VALIDATION or TEST")
    observations = _load_observations(selection)
    references = [row for row in observations if row["role"] == "TRAIN"]
    queries = [row for row in observations if row["role"] == role]
    reference_descriptors = {row["product_id"]: _descriptor(row["raster"]) for row in references}
    samples, retrievals = [], []
    for query in queries:
        ranking = rank_references(_descriptor(query["raster"]), reference_descriptors)
        candidates = []
        selected = None
        for candidate in ranking:
            reference = next(row for row in references if row["product_id"] == candidate.product_id)
            # Held-out acquisition masks are more fragmented than the TRAIN
            # pool.  A 25% common-valid window still supplies >100k exact GT
            # pixels; coverage is recorded and never used to claim full-frame
            # validity.  TRAIN crops retain the builder's stricter 95% rule.
            locations = _crop_candidates(query, reference, 640, minimum_coverage=0.25)
            valid_locations = []
            for location, center in locations:
                source_meta, source_origin = crop_metadata(query["metadata"], center, 640)
                target_meta, target_origin = crop_metadata(reference["metadata"], center, 640)
                source_image, source_valid = _read_window(query["raster"], source_origin, 640)
                target_image, target_valid = _read_window(reference["raster"], target_origin, 640)
                warp, geo_valid = dense_pixel_warp(source_meta, target_meta)
                valid = geo_valid & source_valid & _target_mask_at_warp(target_valid, warp)
                if float(valid.mean()) >= 0.25:
                    valid_locations.append(
                        {
                            "location": location,
                            "center_world_m": [float(center[0]), float(center[1])],
                            "source_origin": list(source_origin),
                            "target_origin": list(target_origin),
                            "warp_ab_px": warp,
                            "valid_ab": valid,
                        }
                    )
            candidates.append(
                {
                    "product_id": reference["product_id"],
                    "rank": candidate.rank,
                    "score": candidate.score,
                    "gt_valid_crops": len(valid_locations),
                }
            )
            if valid_locations and selected is None:
                selected = (reference, valid_locations[0])
        if selected is None:
            raise RuntimeError(
                f"no GT-valid TRAIN reference for held-out {role} query {query['product_id']}"
            )
        reference, crop = selected
        samples.append(
            {
                "sample_id": (
                    f"{role}__{query['product_id']}__{reference['product_id']}__{crop['location']}"
                ),
                "pair_id": f"{query['product_id']}__{reference['product_id']}",
                "source_product_id": query["product_id"],
                "target_product_id": reference["product_id"],
                "source_raster": str(query["raster"]),
                "target_raster": str(reference["raster"]),
                "source_origin": crop["source_origin"],
                "target_origin": crop["target_origin"],
                "center_world_m": crop["center_world_m"],
                "location": crop["location"],
                "warp_ab_px": crop["warp_ab_px"],
                "valid_ab": crop["valid_ab"],
                "gt_valid_coverage": float(np.mean(crop["valid_ab"])),
                "role": role,
            }
        )
        retrievals.append(
            {
                "query_product_id": query["product_id"],
                "role": role,
                "candidates": candidates,
                "selected_reference": reference["product_id"],
                "selection_method": "TRAIN-only histogram retrieval then GT-valid geometry crop",
            }
        )
    return samples, retrievals


def _arrays(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source, _ = _read_window(Path(sample["source_raster"]), tuple(sample["source_origin"]), 640)
    target, _ = _read_window(Path(sample["target_raster"]), tuple(sample["target_origin"]), 640)
    if "gt_path" in sample:
        with np.load(sample["gt_path"]) as payload:
            warp = payload["warp_ab_px"]
            valid = payload["valid_ab"]
    else:
        warp = sample["warp_ab_px"]
        valid = sample["valid_ab"]
    return source, target, np.asarray(warp, dtype=np.float32), np.asarray(valid, dtype=bool)


def _aggregate(errors: list[np.ndarray]) -> dict[str, float | int]:
    values = np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)
    if not values.size:
        raise ValueError("evaluation has no valid correspondence errors")
    return {
        "valid_correspondences": int(values.size),
        "median_epe_px": float(np.median(values)),
        "mean_epe_px": float(np.mean(values)),
        "pck_0.5": float(np.mean(values <= 0.5)),
        "pck_1": float(np.mean(values <= 1.0)),
        "pck_3": float(np.mean(values <= 3.0)),
    }


def _evaluate(
    model: Any, samples: list[dict[str, Any]], preprocess: str, device: torch.device
) -> dict[str, float | int]:
    errors = []
    model.eval()
    for sample in samples:
        source, target, warp, valid = _arrays(sample)
        with torch.inference_mode():
            prediction = model.match(
                as_roma_tensor(preprocess_science(source, preprocess), device),
                as_roma_tensor(preprocess_science(target, preprocess), device),
            )
            predicted = roma_warp_to_pixel(prediction["warp_AB"], (640, 640))[0]
        gt = torch.from_numpy(warp).to(device)
        mask = torch.from_numpy(valid).to(device)
        values = torch.linalg.vector_norm(predicted - gt, dim=-1)[mask].detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite RoMa evaluation output for {sample['sample_id']}")
        errors.append(values)
    return _aggregate(errors)


def _training_batch(
    sample: dict[str, Any], preprocess: str, rng: np.random.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source, target, warp, valid = _arrays(sample)
    source_image = photometric_augment(preprocess_science(source, preprocess), rng)
    target_image = photometric_augment(preprocess_science(target, preprocess), rng)
    return (
        as_roma_tensor(source_image, device),
        as_roma_tensor(target_image, device),
        torch.from_numpy(warp)[None].to(device),
        torch.from_numpy(valid)[None].to(device),
    )


def _load_model(checkpoint: Path) -> tuple[Any, torch.device]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "romav2" / "src"))
    from romav2 import RoMaV2

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = RoMaV2(checkpoint_path=str(checkpoint))
    model.apply_setting("base")
    return model, device


def _monitor_crops(crops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use one deterministic central crop per valid pair at every checkpoint."""
    central = [row for row in crops if row["location"] == "central"]
    by_pair = {row["pair_id"]: row for row in central}
    if len(by_pair) != len({row["pair_id"] for row in crops}):
        raise ValueError("each valid RP-001 pair must provide a central monitor crop")
    return [by_pair[pair_id] for pair_id in sorted(by_pair)]


def _comparison(
    model: Any,
    train_samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[str, dict[str, dict[str, dict[str, float | int]]]]:
    results = {}
    for method in PREPROCESSING_METHODS:
        results[method] = {
            "train_monitor": _evaluate(model, train_samples, method, device),
            "validation": _evaluate(model, validation_samples, method, device),
        }
    selected = max(
        PREPROCESSING_METHODS,
        key=lambda method: (
            float(results[method]["validation"]["pck_1"]),
            float(results[method]["validation"]["pck_0.5"]),
            -float(results[method]["validation"]["median_epe_px"]),
            method == "raw",
        ),
    )
    return selected, results


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.named_parameters()}


def _frozen_changed(before: dict[str, torch.Tensor], model: Any) -> bool:
    now = dict(model.named_parameters())
    return any(
        not torch.equal(value, now[name].detach().cpu())
        for name, value in before.items()
        if not name.startswith("refiners.1.")
    )


def _train_one_lr(
    checkpoint: Path,
    lr: float,
    steps: int,
    samples: list[dict[str, Any]],
    train_monitor: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    preprocess: str,
    output: Path,
) -> dict[str, Any]:
    model, device = _load_model(checkpoint)
    model.eval()
    stride1 = configure_stride1_refiner_training(model)
    before = _snapshot(model)
    optimizer = torch.optim.AdamW(stride1.parameters(), lr=lr)
    ema = RefinerEMA(stride1, decay=0.999)
    pair_counts = Counter(row["pair_id"] for row in samples)
    weights = [float(row["pair_training_weight"]) / pair_counts[row["pair_id"]] for row in samples]
    if any(weight <= 0 or not np.isfinite(weight) for weight in weights):
        raise ValueError("invalid RP-001 weighted-sampling weight")
    random_source = random.Random(20260911)
    np_rng = np.random.default_rng(20260911)
    checkpoints = {10, 25, 50, 100, 250, steps}
    history = []
    output.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        sample = random_source.choices(samples, weights=weights, k=1)[0]
        image_a, image_b, warp, valid = _training_batch(sample, preprocess, np_rng, device)
        model.eval()
        stride1.train()
        result = forward_train(model, image_a, image_b)
        stride1_output = [row for row in result["refiners"] if row["stride"] == 1]
        losses = refiner_stage_losses(stride1_output, warp, valid, (640, 640), (640, 640))
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite stride-1 loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        ema.update(stride1)
        if step in checkpoints:
            model.eval()
            raw_train = _evaluate(model, train_monitor, preprocess, device)
            raw_validation = _evaluate(model, validation_samples, preprocess, device)
            with ema.average_parameters(stride1):
                ema_train = _evaluate(model, train_monitor, preprocess, device)
                ema_validation = _evaluate(model, validation_samples, preprocess, device)
                state_path = output / f"step_{step}_ema_stride1.pt"
                torch.save(stride1.state_dict(), state_path)
            history.append(
                {
                    "step": step,
                    "loss": float(losses["total"].detach().item()),
                    "raw": {"train": raw_train, "validation": raw_validation},
                    "ema": {"train": ema_train, "validation": ema_validation},
                    "state_path": str(state_path),
                }
            )
            print(
                json.dumps(
                    {
                        "lr": lr,
                        "step": step,
                        "loss": float(losses["total"].detach().item()),
                        "validation": ema_validation,
                    }
                ),
                flush=True,
            )
    if _frozen_changed(before, model):
        raise RuntimeError("a frozen official RoMa v2 parameter changed during D1 adaptation")
    if not any(
        not torch.equal(value, dict(model.named_parameters())[name].detach().cpu())
        for name, value in before.items()
        if name.startswith("refiners.1.")
    ):
        raise RuntimeError("D1 adaptation produced no stride-1 parameter update")
    return {"lr": lr, "steps": steps, "checkpoints": history, "frozen_parameters_changed": False}


def _candidate_score(candidate: dict[str, Any], baseline: dict[str, float | int]) -> float:
    validation = candidate["ema"]["validation"]
    if float(validation["pck_1"]) < 0.5 * float(baseline["pck_1"]):
        return -1e9
    return (
        float(validation["pck_1"])
        + 0.25 * float(validation["pck_0.5"])
        - 0.01 * float(validation["median_epe_px"])
        + 0.05 * float(candidate["ema"]["train"]["pck_1"])
    )


def run(selection: Path, run_dir: Path, checkpoint: Path, *, steps: int = 250) -> dict[str, Any]:
    manifest, crops = _load_manifest(run_dir)
    train_monitor = _monitor_crops(crops)
    validation_samples, validation_retrieval = _evaluation_samples(selection, "VALIDATION")
    if len(validation_samples) != 4:
        raise ValueError("expected four held-out RP-001 validation acquisitions")
    model, device = _load_model(checkpoint)
    preprocessing, comparison = _comparison(model, train_monitor, validation_samples, device)
    validation_baseline = comparison[preprocessing]["validation"]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trials = []
    for lr in (1e-6, 3e-6, 1e-5):
        print(f"starting D1 trial at lr={lr:g}", flush=True)
        trials.append(
            _train_one_lr(
                checkpoint,
                lr,
                steps,
                crops,
                train_monitor,
                validation_samples,
                preprocessing,
                run_dir / "checkpoints" / f"lr_{lr:g}",
            )
        )
    candidates = []
    for trial in trials:
        for row in trial["checkpoints"]:
            candidates.append({"lr": trial["lr"], **row})
    selected = max(
        candidates, key=lambda candidate: _candidate_score(candidate, validation_baseline)
    )
    specialist_state = run_dir / "rp001_specialist_stride1.pt"
    specialist_state.write_bytes(Path(selected["state_path"]).read_bytes())
    # TEST is materialized and evaluated exactly once after validation-only selection.
    test_samples, test_retrieval = _evaluation_samples(selection, "TEST")
    if len(test_samples) != 4:
        raise ValueError("expected four held-out RP-001 test acquisitions")
    selected_model, selected_device = _load_model(checkpoint)
    selected_model.refiners["1"].load_state_dict(
        torch.load(specialist_state, map_location=selected_device, weights_only=True)
    )
    test_metrics = _evaluate(selected_model, test_samples, preprocessing, selected_device)
    train_improved = float(selected["ema"]["train"]["pck_1"]) > float(
        comparison[preprocessing]["train_monitor"]["pck_1"]
    )
    validation_usable = float(selected["ema"]["validation"]["pck_1"]) >= 0.5 * float(
        validation_baseline["pck_1"]
    )
    demo_gate = "PASS" if train_improved and validation_usable else "FAIL"
    result = {
        "status": "EXPERIMENTAL_REGION_SPECIALIST",
        "region_id": "RP-001",
        "scope": "Apollo 15 S-IVB Impact only; not production or global lunar evidence",
        "base_model": "official RoMa v2",
        "official_checkpoint_sha256": sha256_file(checkpoint),
        "training": {
            "input_size": [640, 640],
            "trainable_module": "refiners.1 only (D1)",
            "frozen": ["backbone/DINO", "coarse matcher", "stride-4 refiner", "stride-2 refiner"],
            "objective": "corrected source-faithful RoMa robust warp loss, stride-1 only",
            "photometric_augmentation": (
                "TRAIN only: brightness, contrast, gamma; no spatial augmentation"
            ),
            "weighted_sampling": (
                "all GT-valid pairs retained; pair weight divided across its crop samples"
            ),
            "trials": trials,
        },
        "data": {
            "train_acquisitions": manifest["train_acquisition_count"],
            "theoretical_train_pairs": manifest["theoretical_pair_count"],
            "gt_valid_pairs": manifest["gt_valid_pair_count"],
            "rejected_pairs": manifest["rejected_pair_count"],
            "crop_samples": manifest["crop_sample_count"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "preprocessing": {
            "selected": preprocessing,
            "selection_scope": "TRAIN monitor plus all four VALIDATION acquisitions; TEST excluded",
            "comparison": comparison,
        },
        "validation_retrieval": validation_retrieval,
        "selected_checkpoint": {
            "lr": selected["lr"],
            "step": selected["step"],
            "state_path": str(specialist_state),
            "state_sha256": sha256_file(specialist_state),
            "metrics": selected["ema"],
            "selection_rule": (
                "validation PCK@1/PCK@0.5/EPE plus stable train metric; no TEST input"
            ),
        },
        "test": {
            "evaluated_exactly_once_after_freeze": True,
            "metrics": test_metrics,
            "retrieval": test_retrieval,
        },
        "PRODUCTION_GATE": "FAIL",
        "HACKATHON_DEMO_GATE": demo_gate,
        "gate_rationale": {
            "production": (
                "FAIL: regional specialist is deliberately small-sample/region-specific and "
                "does not establish production generalization or complete negatives."
            ),
            "demo": (
                "PASS: D1 train PCK@1 improved and validation retained at least half its official "
                "baseline."
                if demo_gate == "PASS"
                else (
                    "FAIL: real D1 training ran, but the validation-preservation criterion was not "
                    "met."
                )
            ),
        },
    }
    result["result_sha256"] = canonical_json_sha256(result)
    (run_dir / "adaptation_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("configs/regions/rp001.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/rp001_demo_adaptation_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/romav2_official.pt"))
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    print(
        json.dumps(run(args.selection, args.run_dir, args.checkpoint, steps=args.steps), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
