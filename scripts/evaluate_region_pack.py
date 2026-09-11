#!/usr/bin/env python3
"""Evaluate frozen Region Pack queries through retrieval, official RoMa, and geometry."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
import yaml
from rasterio.windows import Window
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.evaluation.protocol import AcceptanceConfig, accept_registration, vrr_far
from chandrappan.geo.footprint import footprint_corners_world
from chandrappan.geometry.verification import VerificationConfig, verify_registration
from chandrappan.region_packs import fractional_zncc_peak, intensity_descriptor, rank_references
from chandrappan.training.coordinates import roma_warp_to_pixel


def _read_crop(path: Path, origin: tuple[int, int], size: int) -> tuple[torch.Tensor, np.ndarray]:
    import rasterio

    with rasterio.open(path) as dataset:
        values = dataset.read(1, window=Window(origin[0], origin[1], size, size)).astype("float32")
    values = np.nan_to_num(values, nan=0.0)
    low, high = np.percentile(values, (1, 99))
    normalized = np.clip((values - low) / (high - low + 1e-6), 0.0, 1.0)
    image = torch.from_numpy(np.repeat(normalized[None], 3, axis=0))[None]
    return image, normalized


def _crop_metadata(
    metadata: Any, center_world: tuple[float, float], size: int
) -> tuple[Any, tuple[int, int]]:
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


def _overlap_center(first: Any, second: Any) -> tuple[float, float]:
    overlap = Polygon(footprint_corners_world(first)).intersection(
        Polygon(footprint_corners_world(second))
    )
    if overlap.is_empty:
        raise ValueError(f"no map overlap for {first.product_id} and {second.product_id}")
    return float(overlap.centroid.x), float(overlap.centroid.y)


def _points(
    prediction: dict[str, torch.Tensor], crop_size: int, *, max_matches: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    warp_ab = prediction["warp_AB"]
    warp_ba = prediction["warp_BA"]
    overlap = prediction["overlap_AB"][0, ..., 0]
    precision = prediction["precision_AB"][0]
    if warp_ba is None:
        raise RuntimeError("production Region Pack requires bidirectional official RoMa output")
    height, width = warp_ab.shape[1:3]
    y, x = torch.meshgrid(
        torch.arange(height, device=warp_ab.device, dtype=warp_ab.dtype) + 0.5,
        torch.arange(width, device=warp_ab.device, dtype=warp_ab.dtype) + 0.5,
        indexing="ij",
    )
    source = torch.stack((x * crop_size / width, y * crop_size / height), dim=-1)
    target = roma_warp_to_pixel(warp_ab, (crop_size, crop_size))[0]
    reverse = torch_f.grid_sample(
        warp_ba.permute(0, 3, 1, 2),
        warp_ab,
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)[0]
    grid = torch.stack((2 * x / width - 1, 2 * y / height - 1), dim=-1)
    cycle_px = torch.linalg.vector_norm(reverse - grid, dim=-1) * crop_size / 2
    precision_eigenvalues = torch.linalg.eigvalsh(precision)
    precision_valid = (precision_eigenvalues[..., 0] > 0) & torch.isfinite(
        precision_eigenvalues
    ).all(dim=-1)
    precision_condition = precision_eigenvalues[..., 1] / precision_eigenvalues[..., 0].clamp_min(
        1e-12
    )
    finite = torch.isfinite(target).all(dim=-1) & torch.isfinite(overlap)
    inside = (
        (target[..., 0] >= 0)
        & (target[..., 0] < crop_size)
        & (target[..., 1] >= 0)
        & (target[..., 1] < crop_size)
    )
    keep = finite & inside & (overlap >= 0.5) & (cycle_px <= 3.0) & precision_valid
    keep &= precision_condition <= 100.0
    if int(keep.sum()) > max_matches:
        scores = torch.where(keep, overlap, torch.tensor(-1.0, device=overlap.device))
        selected = torch.topk(scores.reshape(-1), max_matches).indices
        flat_keep = torch.zeros_like(keep.reshape(-1), dtype=torch.bool)
        flat_keep[selected] = True
        keep = flat_keep.reshape_as(keep)
    diagnostics = {
        "raw_correspondence_count": int(finite.sum()),
        "filtered_correspondence_count": int(keep.sum()),
        "cycle_median_px": float(torch.median(cycle_px[keep]).item())
        if keep.any()
        else float("inf"),
        "precision_condition_median": (
            float(torch.median(precision_condition[keep]).item()) if keep.any() else float("inf")
        ),
    }
    return (
        source[keep].detach().cpu().numpy(),
        target[keep].detach().cpu().numpy(),
        overlap[keep].detach().cpu().numpy(),
        diagnostics,
    )


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ matrix.T
    if matrix.shape == (3, 3):
        projected = projected[:, :2] / projected[:, 2:3]
    return projected


def _ncc_refinement(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    matrix: tuple[tuple[float, ...], ...] | None,
) -> dict[str, float | int | str | None]:
    if matrix is None:
        return {"accepted_control_count": 0, "rejection_reason": "no_geometric_transform"}
    projected = _project(np.asarray(matrix), source)
    controls = []
    patch_radius, search_radius = 12, 5
    for index in np.linspace(0, len(source) - 1, min(100, len(source)), dtype=int):
        sx, sy = np.rint(source[index]).astype(int)
        tx, ty = np.rint(target[index]).astype(int)
        if (
            sx - patch_radius < 0
            or sy - patch_radius < 0
            or sx + patch_radius >= source_image.shape[1]
            or sy + patch_radius >= source_image.shape[0]
            or tx - patch_radius - search_radius < 0
            or ty - patch_radius - search_radius < 0
            or tx + patch_radius + search_radius >= target_image.shape[1]
            or ty + patch_radius + search_radius >= target_image.shape[0]
        ):
            continue
        patch = source_image[
            sy - patch_radius : sy + patch_radius + 1, sx - patch_radius : sx + patch_radius + 1
        ]
        search = target_image[
            ty - patch_radius - search_radius : ty + patch_radius + search_radius + 1,
            tx - patch_radius - search_radius : tx + patch_radius + search_radius + 1,
        ]
        correlation = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
        refinement = fractional_zncc_peak(correlation)
        if not refinement.accepted or refinement.dx_px is None or refinement.dy_px is None:
            continue
        refined = np.array([tx + refinement.dx_px, ty + refinement.dy_px])
        # Reverse refinement rejects one-way local correlations.
        rx, ry = np.rint(refined).astype(int)
        if (
            rx - patch_radius < 0
            or ry - patch_radius < 0
            or rx + patch_radius >= target_image.shape[1]
            or ry + patch_radius >= target_image.shape[0]
            or sx - patch_radius - search_radius < 0
            or sy - patch_radius - search_radius < 0
            or sx + patch_radius + search_radius >= source_image.shape[1]
            or sy + patch_radius + search_radius >= source_image.shape[0]
        ):
            continue
        reverse_patch = target_image[
            ry - patch_radius : ry + patch_radius + 1, rx - patch_radius : rx + patch_radius + 1
        ]
        reverse_search = source_image[
            sy - patch_radius - search_radius : sy + patch_radius + search_radius + 1,
            sx - patch_radius - search_radius : sx + patch_radius + search_radius + 1,
        ]
        reverse = fractional_zncc_peak(
            cv2.matchTemplate(reverse_search, reverse_patch, cv2.TM_CCOEFF_NORMED)
        )
        if not reverse.accepted or reverse.dx_px is None or reverse.dy_px is None:
            continue
        reverse_error = np.linalg.norm(
            np.array([sx + reverse.dx_px, sy + reverse.dy_px]) - source[index]
        )
        if reverse_error > 0.75:
            continue
        controls.append(
            (
                np.linalg.norm(target[index] - projected[index]),
                np.linalg.norm(refined - projected[index]),
            )
        )
    if not controls:
        return {
            "accepted_control_count": 0,
            "rejection_reason": "no_reverse_consistent_zncc_controls",
        }
    values = np.asarray(controls)
    pre_ncc = float(np.median(values[:, 0]))
    post_ncc = float(np.median(values[:, 1]))
    if post_ncc > pre_ncc:
        return {
            "proposed_control_count": int(len(values)),
            "accepted_control_count": 0,
            "pre_ncc_median_px": pre_ncc,
            "post_ncc_median_px": post_ncc,
            "rejection_reason": "ncc_degrades_geometric_residual",
        }
    return {
        "accepted_control_count": int(len(values)),
        "pre_ncc_median_px": pre_ncc,
        "post_ncc_median_px": post_ncc,
        "rejection_reason": None,
    }


def _correspondence_metrics(
    source: np.ndarray, target: np.ndarray, source_metadata: Any, target_metadata: Any
) -> tuple[dict[str, float | int] | None, np.ndarray]:
    """Measure filtered official-RoMa matches against map-derived dense GT."""
    if not len(source):
        return None, np.empty(0, dtype=np.float64)
    world_x, world_y = source_metadata.transform.pixel_to_world(source[:, 0], source[:, 1])
    gt_x, gt_y = target_metadata.transform.world_to_pixel(world_x, world_y)
    valid = (
        (gt_x >= 0) & (gt_x < target_metadata.width) & (gt_y >= 0) & (gt_y < target_metadata.height)
    )
    errors = np.hypot(target[:, 0] - gt_x, target[:, 1] - gt_y)[valid]
    if not len(errors):
        return None, errors
    return (
        {
            "valid_correspondences": int(len(errors)),
            "median_epe_px": float(np.median(errors)),
            "mean_epe_px": float(np.mean(errors)),
            "p90_epe_px": float(np.quantile(errors, 0.90)),
            "p95_epe_px": float(np.quantile(errors, 0.95)),
            "p99_epe_px": float(np.quantile(errors, 0.99)),
            **{
                f"pck_{threshold:g}": float(np.mean(errors <= threshold))
                for threshold in (0.5, 1, 2, 3, 5)
            },
        },
        errors,
    )


def _load_observations(root: Path) -> list[dict[str, Any]]:
    import csv

    with (root / "observation_catalog.csv").open(newline="", encoding="utf-8") as handle:
        catalog = list(csv.DictReader(handle))
    result = []
    for row in catalog:
        metadata_path = root / "metadata" / f"{row['product_id']}.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        from chandrappan.data.lroc_ingest import read_lroc_metadata

        result.append(
            {
                **row,
                "metadata": read_lroc_metadata(
                    payload["raw_geotiff"], Path(payload["raw_geotiff"]).with_suffix(".xml")
                ),
                "raster": Path(payload["processed_science_raster"]),
            }
        )
    return result


def _descriptor(path: Path) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        image = dataset.read(1, out_shape=(1, 256, 256), resampling=Resampling.average)
    return intensity_descriptor(image)


def evaluate(selection: Path, checkpoint: Path, *, top_k: int, crop_size: int) -> dict[str, object]:
    config = yaml.safe_load(selection.read_text(encoding="utf-8"))
    root = Path(config["products"][0]["local_path"]).parents[1]
    observations = _load_observations(root)
    references = [row for row in observations if row["role"] == "TRAIN"]
    queries = [row for row in observations if row["role"] in {"VALIDATION", "TEST"}]
    if top_k < 3:
        raise ValueError("production Region Pack evaluation requires top_k >= 3")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "romav2" / "src"))
    from romav2 import RoMaV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoMaV2(checkpoint_path=str(checkpoint))
    model.apply_setting("precise")
    if not model.bidirectional:
        raise RuntimeError("official RoMa precise setting did not enable bidirectional inference")
    acceptance_payload = yaml.safe_load(Path("configs/registration_acceptance.yaml").read_text())
    acceptance = AcceptanceConfig(**acceptance_payload["acceptance"])
    reference_descriptors = {}
    for reference in references:
        reference_descriptors[reference["product_id"]] = _descriptor(reference["raster"])
    rows = []
    for query in queries:
        ranking = rank_references(_descriptor(query["raster"]), reference_descriptors)
        candidates = []
        for retrieved in ranking[:top_k]:
            reference = next(row for row in references if row["product_id"] == retrieved.product_id)
            center = _overlap_center(query["metadata"], reference["metadata"])
            query_meta, query_origin = _crop_metadata(query["metadata"], center, crop_size)
            ref_meta, ref_origin = _crop_metadata(reference["metadata"], center, crop_size)
            image_query, array_query = _read_crop(query["raster"], query_origin, crop_size)
            image_ref, array_ref = _read_crop(reference["raster"], ref_origin, crop_size)
            with torch.inference_mode():
                prediction = model.match(image_query.to(device), image_ref.to(device))
            source, target, _, matching = _points(prediction, crop_size, max_matches=4000)
            correspondence, errors = _correspondence_metrics(source, target, query_meta, ref_meta)
            verification = verify_registration(
                source, target, (crop_size, crop_size), config=VerificationConfig()
            )
            metrics = verification.selected.metrics() if verification.selected else None
            accepted = bool(
                metrics and verification.accepted and accept_registration(metrics, acceptance)
            )
            ncc = _ncc_refinement(
                array_query,
                array_ref,
                source,
                target,
                verification.selected.matrix if verification.selected else None,
            )
            candidates.append(
                {
                    "reference_product_id": reference["product_id"],
                    "retrieval_rank": retrieved.rank,
                    "retrieval_score": retrieved.score,
                    "matching": matching,
                    "correspondence_metrics": correspondence,
                    "correspondence_errors_px": errors.tolist(),
                    "verification": verification.as_dict(),
                    "accepted_geometry": accepted,
                    "ncc": ncc,
                    "query_crop": str(query["raster"]),
                    "reference_crop": str(reference["raster"]),
                }
            )
        selected = next((row for row in candidates if row["accepted_geometry"]), candidates[0])
        verdict = "VERIFIED" if selected["accepted_geometry"] else "REJECTED"
        rows.append(
            {
                "query_product_id": query["product_id"],
                "query_role": query["role"],
                "held_out": query["role"] == "TEST",
                "candidate_ranking": candidates,
                "selected_reference": selected["reference_product_id"],
                "selection_method": "regional_reference_bank_then_geometry",
                "verdict": verdict,
                "metrics": selected["verification"].get("selected", {}),
                "correspondence_metrics": selected["correspondence_metrics"],
                "ncc": selected["ncc"],
            }
        )
    test_rows = [row for row in rows if row["query_role"] == "TEST"]
    negative_path = Path("results/RP-001_hard_negative_suite.json")
    negative_suite = (
        json.loads(negative_path.read_text(encoding="utf-8")) if negative_path.exists() else None
    )
    negative_outcomes = (
        [(False, bool(row["accepted"])) for row in negative_suite["cases"]]
        if negative_suite is not None
        else []
    )
    outcomes = [(True, row["verdict"] == "VERIFIED") for row in test_rows] + negative_outcomes
    test_errors = (
        np.concatenate(
            [
                np.asarray(
                    row["candidate_ranking"][0]["correspondence_errors_px"], dtype=np.float64
                )
                for row in test_rows
                if row["candidate_ranking"][0]["correspondence_errors_px"]
            ]
        )
        if any(row["candidate_ranking"][0]["correspondence_errors_px"] for row in test_rows)
        else np.empty(0)
    )
    dense_metrics = (
        {
            "valid_correspondences": int(len(test_errors)),
            "median_epe_px": float(np.median(test_errors)),
            "mean_epe_px": float(np.mean(test_errors)),
            "p90_epe_px": float(np.quantile(test_errors, 0.90)),
            "p95_epe_px": float(np.quantile(test_errors, 0.95)),
            "p99_epe_px": float(np.quantile(test_errors, 0.99)),
            **{
                f"pck_{threshold:g}": float(np.mean(test_errors <= threshold))
                for threshold in (0.5, 1, 2, 3, 5)
            },
        }
        if len(test_errors)
        else None
    )
    result = {
        "region_id": "RP-001",
        "matcher": "official RoMa v2, precise bidirectional setting",
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "reference_bank_role": "TRAIN only",
        "top_k": top_k,
        "query_count": len(rows),
        "test_query_count": len(test_rows),
        "retrieval": {
            "top_1_region_family": 1.0,
            "top_3_region_family": 1.0,
            "top_5_region_family": 1.0,
            "note": (
                "All eligible references belong to RP-001; this reports family retrieval, not "
                "visual match correctness."
            ),
        },
        "registration": vrr_far(outcomes),
        "dense_matching": {
            "scope": (
                "top-ranked candidate's filtered, overlap/cycle/precision-valid correspondences"
            ),
            "metrics": dense_metrics,
        },
        "negative_rejection": (
            {
                "path": str(negative_path),
                "evaluated_categories": negative_suite["available_negative_categories"],
                "unavailable_required_category": negative_suite["unavailable_required_category"],
            }
            if negative_suite is not None
            else "not_evaluated_by_this_positive_query_run"
        ),
        "acceptance_status": (
            "INCOMPLETE: hard-negative evaluation is required before RP-001 can pass."
        ),
        "queries": rows,
    }
    output_json = Path("results/RP-001_official_romav2_baseline.json")
    output_md = Path("results/RP-001_official_romav2_baseline.md")
    output_json.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    summary = (
        "# RP-001 official RoMa v2 baseline\n\n"
        f"- Held-out test queries: {len(test_rows)}\n"
        f"- Verification rate on positive held-out queries: {result['registration']['vrr']}\n"
        "- Filtered dense-match median EPE: "
        f"{dense_metrics['median_epe_px'] if dense_metrics else 'n/a'}\n"
        f"- FAR: {result['registration']['far']}\n"
        "- Status: incomplete: required semantic-similarity/shadow negatives and positive "
        "verification improvements remain outstanding.\n"
    )
    output_md.write_text(
        summary,
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("configs/regions/rp001.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/romav2_official.pt"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=640)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.selection, args.checkpoint, top_k=args.top_k, crop_size=args.crop_size)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
