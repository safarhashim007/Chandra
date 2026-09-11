#!/usr/bin/env python3
"""Render transparent offline showcase assets for the RP-001 specialist demo."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.manifest import read_manifest
from chandrappan.evaluation.metrics import correspondence_metrics
from chandrappan.geometry.verification import VerificationConfig, verify_registration
from chandrappan.rp001_demo import (
    as_roma_tensor,
    canonical_json_sha256,
    preprocess_science,
    sha256_file,
)
from chandrappan.training.coordinates import roma_warp_to_pixel
from scripts.run_rp001_demo_adaptation import (
    _arrays,
    _evaluation_samples,
    _load_manifest,
    _load_model,
    _monitor_crops,
)


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float32)))
    if matrix.shape == (2, 3):
        return homogeneous @ matrix.T
    values = homogeneous @ matrix.T
    return values[:, :2] / values[:, 2:3]


def _distributed(points: np.ndarray, scores: np.ndarray, limit: int = 96) -> np.ndarray:
    """Return a high-score, grid-distributed subset for readable presentation."""
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(points):
        bucket = (min(5, int(point[0] * 6 / 640)), min(5, int(point[1] * 6 / 640)))
        buckets.setdefault(bucket, []).append(index)
    for indices in buckets.values():
        indices.sort(key=lambda index: (-float(scores[index]), index))
    selected: list[int] = []
    while len(selected) < limit and any(buckets.values()):
        for bucket in sorted(buckets):
            if buckets[bucket] and len(selected) < limit:
                selected.append(buckets[bucket].pop(0))
    return np.asarray(selected, dtype=int)


def _match_summary(
    model: Any,
    sample: dict[str, Any],
    preprocess: str,
    device: torch.device,
    *,
    has_gt: bool,
) -> dict[str, Any]:
    source_image, target_image, warp_gt, valid_gt = _arrays(sample)
    with torch.inference_mode():
        prediction = model.match(
            as_roma_tensor(preprocess_science(source_image, preprocess), device),
            as_roma_tensor(preprocess_science(target_image, preprocess), device),
        )
    warp = roma_warp_to_pixel(prediction["warp_AB"], (640, 640))[0].detach().cpu().numpy()
    confidence = prediction["overlap_AB"][0, ..., 0].detach().cpu().numpy()
    yy, xx = np.indices(warp.shape[:2], dtype=np.float32)
    source = np.stack((xx + 0.5, yy + 0.5), axis=-1)
    finite = np.isfinite(warp).all(axis=-1) & np.isfinite(confidence)
    inside = (warp[..., 0] >= 0) & (warp[..., 0] < 640) & (warp[..., 1] >= 0) & (warp[..., 1] < 640)
    keep = finite & inside & (confidence >= 0.5)
    metrics = None
    if has_gt:
        gt = torch.from_numpy(warp_gt)[None].to(device)
        valid = torch.from_numpy(valid_gt)[None].to(device)
        predicted = torch.from_numpy(warp)[None].to(device)
        metrics = correspondence_metrics(predicted, gt, valid, (0.5, 1.0, 3.0))
        keep &= valid_gt
    flat = np.flatnonzero(keep)
    if len(flat) > 4000:
        flat = flat[np.argsort(-confidence.reshape(-1)[flat], kind="stable")[:4000]]
    match_source = source.reshape(-1, 2)[flat]
    match_target = warp.reshape(-1, 2)[flat]
    match_scores = confidence.reshape(-1)[flat]
    verification = verify_registration(
        match_source, match_target, (640, 640), config=VerificationConfig()
    )
    inliers = np.zeros(len(match_source), dtype=bool)
    if verification.selected and verification.selected.matrix is not None:
        projected = _project(np.asarray(verification.selected.matrix), match_source)
        inliers = np.linalg.norm(projected - match_target, axis=1) <= 3.0
    presentable = np.flatnonzero(inliers)
    if not len(presentable):
        presentable = np.arange(len(match_source))
    displayed = _distributed(match_source[presentable], match_scores[presentable])
    displayed = presentable[displayed]
    return {
        "source_image": preprocess_science(source_image, preprocess),
        "target_image": preprocess_science(target_image, preprocess),
        "warp": warp,
        "source": match_source,
        "target": match_target,
        "scores": match_scores,
        "inliers": inliers,
        "displayed": displayed,
        "verification": verification.as_dict(),
        "metrics": metrics,
        "raw_correspondences": int(len(match_source)),
        "inlier_count": int(inliers.sum()),
    }


def _text(canvas: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(
        canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA
    )


def _metric_text(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    pck = f"PCK@1 {metrics['pck_1']:.3f}" if metrics else "PCK@1 n/a"
    return (
        f"matches {summary['raw_correspondences']} | inliers {summary['inlier_count']} | "
        f"{pck} | geometry {'VERIFIED' if summary['verification']['accepted'] else 'REJECTED'}"
    )


def _render_matches(summary: dict[str, Any], label: str, destination: Path) -> None:
    first = np.rint(summary["source_image"] * 255).astype(np.uint8)
    second = np.rint(summary["target_image"] * 255).astype(np.uint8)
    panel = np.zeros((714, 1280, 3), dtype=np.uint8)
    panel[74:, :640] = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    panel[74:, 640:] = cv2.cvtColor(second, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(panel, (0, 0), (1279, 73), (20, 29, 42), thickness=-1)
    _text(panel, label, (16, 28), 0.75)
    _text(panel, _metric_text(summary), (16, 54), 0.48)
    colors = [(68, 220, 255), (99, 255, 126), (255, 173, 71), (255, 108, 190)]
    for number, index in enumerate(summary["displayed"], start=1):
        source = np.rint(summary["source"][index]).astype(int)
        target = np.rint(summary["target"][index]).astype(int)
        color = colors[(number - 1) % len(colors)]
        a = (int(source[0]), int(source[1] + 74))
        b = (int(target[0] + 640), int(target[1] + 74))
        cv2.line(panel, a, b, color, 1, cv2.LINE_AA)
        cv2.circle(panel, a, 4, color, 1, cv2.LINE_AA)
        cv2.circle(panel, b, 4, color, 1, cv2.LINE_AA)
        if number <= 12:
            cv2.putText(
                panel,
                f"M{number:02d}",
                (a[0] + 6, a[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
            )
            cv2.putText(
                panel,
                f"M{number:02d}",
                (b[0] + 6, b[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), panel):
        raise RuntimeError(f"failed to write match asset: {destination}")


def _render_alignment(summary: dict[str, Any], prefix: Path) -> dict[str, str]:
    first = np.rint(summary["source_image"] * 255).astype(np.uint8)
    second = np.rint(summary["target_image"] * 255).astype(np.uint8)
    warp = summary["warp"].astype(np.float32)
    registered = cv2.remap(second, warp[..., 0], warp[..., 1], cv2.INTER_LINEAR, borderValue=0)
    overlay = cv2.addWeighted(first, 0.5, registered, 0.5, 0)
    difference = cv2.absdiff(first, registered)
    paths = {
        "source": prefix.with_name(prefix.name + "_source.png"),
        "registered": prefix.with_name(prefix.name + "_registered.png"),
        "overlay": prefix.with_name(prefix.name + "_overlay.png"),
        "difference": prefix.with_name(prefix.name + "_difference.png"),
        "blink": prefix.with_name(prefix.name + "_blink.gif"),
    }
    for key in ("source", "registered", "overlay", "difference"):
        image = {
            "source": first,
            "registered": registered,
            "overlay": overlay,
            "difference": difference,
        }[key]
        if not cv2.imwrite(str(paths[key]), image):
            raise RuntimeError(f"failed to write alignment asset: {paths[key]}")
    from PIL import Image

    Image.fromarray(first).save(
        paths["blink"],
        save_all=True,
        append_images=[Image.fromarray(registered)],
        duration=500,
        loop=0,
    )
    return {key: str(path) for key, path in paths.items()}


def _render_comparison(
    official: dict[str, Any], specialist: dict[str, Any], title: str, destination: Path
) -> None:
    temporary_official = destination.with_name(destination.stem + "_official.png")
    temporary_specialist = destination.with_name(destination.stem + "_specialist.png")
    _render_matches(official, f"{title} — Official RoMa v2", temporary_official)
    _render_matches(specialist, f"{title} — RP-001 Specialist (experimental)", temporary_specialist)
    top = cv2.imread(str(temporary_official), cv2.IMREAD_COLOR)
    bottom = cv2.imread(str(temporary_specialist), cv2.IMREAD_COLOR)
    if top is None or bottom is None:
        raise RuntimeError("failed to reload rendered comparison image")
    if not cv2.imwrite(str(destination), cv2.vconcat((top, bottom))):
        raise RuntimeError(f"failed to write comparison asset: {destination}")


def _external_negative(sample: dict[str, Any]) -> dict[str, Any]:
    records = read_manifest("data/expanded/images.parquet")
    record = next(row for row in records if row.region_id != "E009S3481")
    import rasterio

    with rasterio.open(record.image_path) as dataset:
        mask = dataset.read_masks(1) > 0
        integral = np.pad(mask.astype(np.uint32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = (
            integral[640:, 640:]
            - integral[:-640, 640:]
            - integral[640:, :-640]
            + integral[:-640, :-640]
        )
        y, x = np.unravel_index(np.argmax(sums), sums.shape)
    result = dict(sample)
    result.update(
        {
            "sample_id": "hard_negative_other_lunar_region",
            "target_product_id": record.product_id,
            "target_raster": record.image_path,
            "target_origin": [int(x), int(y)],
        }
    )
    return result


def _case_score(official: dict[str, Any], specialist: dict[str, Any]) -> float:
    if official["metrics"] is None or specialist["metrics"] is None:
        return -np.inf
    return float(specialist["metrics"]["pck_1"]) - float(official["metrics"]["pck_1"])


def _showcase_key(
    sample: dict[str, Any], official: dict[str, Any], specialist: dict[str, Any]
) -> tuple[bool, bool, float, float, float]:
    """Prefer readable, genuinely geometry-verified held-out showcase imagery."""
    metrics = specialist["metrics"]
    if metrics is None:
        return (False, False, 0.0, -np.inf, -np.inf)
    return (
        bool(specialist["verification"]["accepted"]),
        bool(official["verification"]["accepted"]),
        float(sample.get("gt_valid_coverage", 1.0)),
        float(metrics["pck_1"]),
        _case_score(official, specialist),
    )


def _html(status: str, train: str, validation: str, provenance: str) -> str:
    # The static page intentionally remains a real precomputed fallback. It
    # never claims to run inference and is regenerated from tracked source.
    del status, train, validation, provenance
    return Path("frontend/static_fallback.html").read_text(encoding="utf-8")


def build(selection: Path, run_dir: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    result = json.loads((run_dir / "adaptation_result.json").read_text(encoding="utf-8"))
    paired_path = run_dir / "paired_evaluation.json"
    paired_payload = (
        json.loads(paired_path.read_text(encoding="utf-8")) if paired_path.exists() else None
    )
    paired_summary = (
        {
            split: {
                model: {
                    **values["dense_gt"],
                    "vrr": values["positive_vrr"]["vrr"],
                    "accepted": values["positive_vrr"]["accepted"],
                    "positive_queries": values["positive_vrr"]["positive_queries"],
                }
                for model, values in comparison.items()
            }
            for split, comparison in paired_payload["comparison"].items()
        }
        if paired_payload is not None
        else None
    )
    manifest, crops = _load_manifest(run_dir)
    preprocessing = result["preprocessing"]["selected"]
    train_samples = _monitor_crops(crops)
    validation_samples, validation_retrieval = _evaluation_samples(selection, "VALIDATION")
    model, device = _load_model(checkpoint)
    official_stride1 = copy.deepcopy(model.refiners["1"].state_dict())
    specialist = torch.load(
        run_dir / "rp001_specialist_stride1.pt", map_location=device, weights_only=True
    )

    def compare(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        model.refiners["1"].load_state_dict(official_stride1)
        official = _match_summary(model, sample, preprocessing, device, has_gt=True)
        model.refiners["1"].load_state_dict(specialist)
        specialist_summary = _match_summary(model, sample, preprocessing, device, has_gt=True)
        return official, specialist_summary

    train_cases = [(sample, *compare(sample)) for sample in train_samples]
    validation_cases = [(sample, *compare(sample)) for sample in validation_samples]
    train_sample, train_official, train_specialist = max(
        train_cases, key=lambda row: _showcase_key(row[0], row[1], row[2])
    )
    validation_sample, validation_official, validation_specialist = max(
        validation_cases, key=lambda row: _showcase_key(row[0], row[1], row[2])
    )
    output.mkdir(parents=True, exist_ok=True)
    _render_comparison(
        train_official,
        train_specialist,
        "TRAINING-DOMAIN EXAMPLE",
        output / "official_vs_specialist_matches.png",
    )
    _render_comparison(
        validation_official,
        validation_specialist,
        "HELD-OUT ACQUISITION - NOT USED FOR TRAINING",
        output / "heldout_validation_comparison.png",
    )
    train_alignment = _render_alignment(train_specialist, output / "training_domain")
    validation_alignment = _render_alignment(validation_specialist, output / "heldout_validation")
    model.refiners["1"].load_state_dict(specialist)
    negative = _match_summary(
        model, _external_negative(train_sample), preprocessing, device, has_gt=False
    )
    _render_matches(negative, "HARD NEGATIVE — OTHER LUNAR REGION", output / "hard_negative.png")
    contact_sheet = Path("demo/RP-001/train_contact_sheet.png")
    if not contact_sheet.exists():
        raise FileNotFoundError("RP-001 TRAIN contact sheet is required for demo provenance")
    shutil.copy2(contact_sheet, output / "training_data_contact_sheet.png")
    selected_pairs = {
        "valid_pair_count": manifest["gt_valid_pair_count"],
        "crop_sample_count": manifest["crop_sample_count"],
        "pairs": manifest["valid_pairs"],
        "rejected_pairs": manifest["rejected_pairs"],
    }
    (output / "selected_training_pairs.json").write_text(
        json.dumps(selected_pairs, indent=2) + "\n", encoding="utf-8"
    )
    mac_root = output / "mac_inference"
    (mac_root / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, mac_root / "models/romav2_official.pt")
    mac_config = {
        "mode": "inference_only",
        "network": "disabled_after_bundle_creation",
        "device_priority": ["mps", "cpu"],
        "cuda_required": False,
        "live_inference_entrypoint": "scripts/run_rp001_live_demo.py",
        "official_checkpoint": "models/romav2_official.pt",
        "production_matcher": "Official RoMa v2 (frozen)",
        "d1_status": "EXPERIMENTAL · NOT PROMOTED · NEVER LOADED BY LIVE DEMO",
        "live_demo_resolution": 640,
        "input": "local map-projected LROC science raster plus XML metadata",
        "reference_bank": "12 cached TRAIN descriptors and thumbnails, loaded locally at startup",
        "caveat": (
            "Portable bundle prepared on Linux; MPS execution must be smoke-tested on the "
            "presentation Mac before use."
        ),
    }
    (output / "mac_demo_config.json").write_text(
        json.dumps(mac_config, indent=2) + "\n", encoding="utf-8"
    )
    data_card = {
        "status": "EXPERIMENTAL_REGION_SPECIALIST",
        "checkpoint_path": str(run_dir / "rp001_specialist_stride1.pt"),
        "base_model": result["base_model"],
        "official_romav2_checkpoint_hash": result["official_checkpoint_sha256"],
        "git_commit": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], text=True)
        .strip(),
        "train_acquisitions": manifest["train_acquisition_count"],
        "valid_pair_count": manifest["gt_valid_pair_count"],
        "crop_sample_count": manifest["crop_sample_count"],
        "preprocessing": preprocessing,
        "objective": result["training"]["objective"],
        "learning_rate": result["selected_checkpoint"]["lr"],
        "steps": result["selected_checkpoint"]["step"],
        "training_metrics": result["selected_checkpoint"]["metrics"]["train"],
        "validation_metrics": result["selected_checkpoint"]["metrics"]["validation"],
        "test_metrics_after_final_freeze": result["test"]["metrics"],
        "manifest_sha256": manifest["manifest_sha256"],
        "PRODUCTION_GATE": result["PRODUCTION_GATE"],
        "HACKATHON_DEMO_GATE": result["HACKATHON_DEMO_GATE"],
    }
    (output / "rp001_specialist_checkpoint.json").write_text(
        json.dumps(data_card, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "region_id": "RP-001",
        "training_data": {
            "observations": manifest["train_acquisition_count"],
            "valid_pairs": manifest["gt_valid_pair_count"],
            "crops": manifest["crop_sample_count"],
            "validation_used_for_training": 0,
            "test_used_for_training": 0,
        },
        "auto_reference_selection": validation_retrieval,
        "showcases": {
            "training_domain": {
                "sample_id": train_sample["sample_id"],
                "official": train_official["metrics"],
                "specialist": train_specialist["metrics"],
                "metric_delta_pck_1": _case_score(train_official, train_specialist),
                "alignment_assets": train_alignment,
            },
            "heldout_validation": {
                "sample_id": validation_sample["sample_id"],
                "official": validation_official["metrics"],
                "specialist": validation_specialist["metrics"],
                "metric_delta_pck_1": _case_score(validation_official, validation_specialist),
                "alignment_assets": validation_alignment,
            },
            "hard_negative": {
                "verification": negative["verification"],
                "accepted_geometry": negative["verification"]["accepted"],
            },
        },
        "gates": {
            "PRODUCTION_GATE": result["PRODUCTION_GATE"],
            "HACKATHON_DEMO_GATE": result["HACKATHON_DEMO_GATE"],
        },
    }
    (output / "provenance_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(
        _html(
            result["HACKATHON_DEMO_GATE"],
            "training_domain_overlay.png",
            "heldout_validation_overlay.png",
            "training_data_contact_sheet.png",
        ),
        encoding="utf-8",
    )
    report = {
        "region_id": "RP-001",
        "status": result["HACKATHON_DEMO_GATE"],
        "data_card": data_card,
        "paired_base640_comparison": paired_summary,
        "provenance": provenance,
        "assets": {
            path.name: sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file()
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    Path("results/RP-001_DEMO_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    markdown = "\n".join(
        [
            "# RP-001 hackathon demo report",
            "",
            "## Status",
            "",
            f"- EXPERIMENTAL REGION-SPECIFIC ADAPTATION: `{result['HACKATHON_DEMO_GATE']}`",
            f"- RP-001 PRODUCTION GATE: `{result['PRODUCTION_GATE']}`",
            "- Scope: Apollo 15 S-IVB Impact only; not evidence of global lunar generalization.",
            "",
            "## Training provenance",
            "",
            (
                f"- 12 TRAIN acquisitions → {manifest['gt_valid_pair_count']} GT-valid pairs → "
                f"{manifest['crop_sample_count']} 640×640 crops."
            ),
            "- Validation used for training: 0; TEST used for training: 0.",
            f"- Selected preprocessing: `{preprocessing}` (TRAIN/VALIDATION comparison only).",
            (
                f"- Selected D1 checkpoint: LR `{result['selected_checkpoint']['lr']}`, "
                f"step `{result['selected_checkpoint']['step']}`."
            ),
            "",
            "## Outcome",
            "",
            (
                "- The real D1 run completed, but it did not improve monitored TRAIN PCK@1 or "
                "preserve the selected validation criterion strongly enough for a demo PASS."
            ),
            (
                "- TEST metrics were recorded after checkpoint freeze and were not used for "
                "selection."
            ),
            (
                "- Static assets are labeled precomputed examples. The Mac profile is "
                "inference-only and requires an MPS smoke test on the presentation device."
            ),
            "",
        ]
    )
    if paired_summary is not None:
        rows = [
            "## Paired base-640 metrics",
            "",
            "| Split | Model | PCK@0.5 | PCK@1 | Median EPE (px) | VRR |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for split in ("TRAIN", "VALIDATION", "TEST"):
            for model, label in (("official_romav2", "Official"), ("rp001_d1", "D1")):
                metric = paired_summary[split][model]
                rows.append(
                    f"| {split} | {label} | {metric['pck_0.5']:.6f} | "
                    f"{metric['pck_1']:.6f} | {metric['median_epe_px']:.6f} | "
                    f"{metric['accepted']}/{metric['positive_queries']} ({metric['vrr']:.6f}) |"
                )
        markdown += "\n".join(["", *rows, ""])
    Path("results/RP-001_DEMO_REPORT.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("configs/regions/rp001.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/rp001_demo_adaptation_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/romav2_official.pt"))
    parser.add_argument("--output", type=Path, default=Path("demo/RP-001/final"))
    args = parser.parse_args()
    print(json.dumps(build(args.selection, args.run_dir, args.checkpoint, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
