#!/usr/bin/env python3
"""Assemble the relocatable, offline Chandrappan Mac presentation bundle.

This intentionally copies only the official RoMa v2, processed RP-001 assets,
cached descriptors/thumbnails, source, and a small parity fixture.  Raw PDS3,
training crops, GT archives, optimizer states, CUDA environments, and caches
are excluded by construction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.gt_warp import dense_pixel_warp
from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.region_packs import intensity_descriptor
from chandrappan.rp001_demo import canonical_json_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGION_ROOT = ROOT / "data" / "region_packs" / "RP-001"
CHECKPOINT = ROOT / "checkpoints" / "romav2_official.pt"
DINO_HUB_SOURCE = (
    Path.home()
    / ".cache"
    / "torch"
    / "hub"
    / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(
    source: Path, destination: Path, *, ignore: shutil.IgnorePattern | None = None
) -> None:
    shutil.copytree(source, destination, ignore=ignore, dirs_exist_ok=True)


def _descriptor(path: Path) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        image = dataset.read(1, out_shape=(1, 256, 256), resampling=Resampling.average)
    return intensity_descriptor(image)


def _metadata_payload(metadata: Any, source_path: str) -> dict[str, Any]:
    payload = asdict(metadata)
    payload["source_path"] = source_path
    return payload


def _target_mask_at_warp(target_mask: np.ndarray, warp: np.ndarray) -> np.ndarray:
    x = np.rint(warp[..., 0] - 0.5).astype(int)
    y = np.rint(warp[..., 1] - 0.5).astype(int)
    inside = (x >= 0) & (x < target_mask.shape[1]) & (y >= 0) & (y < target_mask.shape[0])
    result = np.zeros_like(inside, dtype=bool)
    result[inside] = target_mask[y[inside], x[inside]]
    return result


def _build_hard_negative(destination: Path) -> str:
    import rasterio

    candidates = sorted((ROOT / "data" / "raw" / "lroc_corpus").glob("*.TIF"))
    source = next((path for path in candidates if "E009S3481" not in path.name), None)
    if source is None:
        raise FileNotFoundError(
            "no external lunar raster is available for the hard-negative fixture"
        )
    with rasterio.open(source) as dataset:
        mask = dataset.read_masks(1) > 0
        size = 640
        integral = np.pad(mask.astype(np.uint32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )
        y, x = np.unravel_index(np.argmax(sums), sums.shape)
        values = dataset.read(1, window=((int(y), int(y + size)), (int(x), int(x + size))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, values.astype(np.float32))
    return sha256_file(destination)


def _parity_fixture(
    destination: Path,
    source_record: dict[str, Any],
    target_record: dict[str, Any],
    origin: tuple[int, int],
) -> None:
    import rasterio
    from rasterio.windows import Window

    source_meta = source_record["metadata"]
    target_meta = target_record["metadata"]
    source_crop = source_meta
    target_crop = target_meta
    from dataclasses import replace

    world = source_meta.transform.pixel_to_world(*origin)
    source_crop = replace(
        source_meta,
        width=640,
        height=640,
        transform=replace(
            source_meta.transform, x_origin=float(world[0]), y_origin=float(world[1])
        ),
    )
    target_crop = replace(
        target_meta,
        width=640,
        height=640,
        transform=replace(
            target_meta.transform, x_origin=float(world[0]), y_origin=float(world[1])
        ),
    )
    warp, geometry_valid = dense_pixel_warp(source_crop, target_crop)
    with (
        rasterio.open(source_record["raster"]) as source_ds,
        rasterio.open(target_record["raster"]) as target_ds,
    ):
        source_mask = source_ds.read_masks(1, window=Window(*origin, 640, 640)) > 0
        target_mask = target_ds.read_masks(1, window=Window(*origin, 640, 640)) > 0
    valid = geometry_valid & source_mask & _target_mask_at_warp(target_mask, warp)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, warp_ab_px=warp.astype(np.float32), valid_ab=valid)


def _purpose(path: Path) -> str:
    if path.parts[0] == "models":
        return "official frozen RoMa v2 inference weights"
    if path.parts[0] == "data":
        return "processed RP-001 inference asset"
    if path.parts[0] == "frontend":
        return "offline presentation frontend"
    if path.parts[0] == "vendor":
        return "bundled official RoMa v2 source"
    if path.parts[0] == "artifacts":
        return "writable local live-demo output directory"
    return "Mac demo runtime source or configuration"


def _write_transfer_manifest(bundle: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        relative = path.relative_to(bundle)
        if relative.name == "TRANSFER_MANIFEST.md":
            continue
        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "purpose": _purpose(relative),
            }
        )
    summary = {
        "bundle": str(bundle),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "model_files": [row for row in files if row["path"].startswith("models/")],
        "rp001_files": [row for row in files if row["path"].startswith("data/rp001/")],
    }
    lines = [
        "# Chandrappan Mac M4 transfer manifest",
        "",
        f"- Files: `{summary['file_count']}`",
        f"- Total bytes: `{summary['total_bytes']}`",
        "- Network required after Mac setup: `NO`",
        "- Excluded by construction: raw PDS3, GT archive, training crops, optimizer state, "
        "CUDA environment, NVIDIA tools, package caches.",
        "",
        "## Files",
        "",
        "| Relative path | Bytes | SHA256 | Purpose |",
        "|---|---:|---|---|",
    ]
    for row in files:
        lines.append(f"| `{row['path']}` | {row['bytes']} | `{row['sha256']}` | {row['purpose']} |")
    (bundle / "TRANSFER_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build(destination: Path, *, force: bool = False) -> dict[str, Any]:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"official checkpoint is missing: {CHECKPOINT}")
    if not (DINO_HUB_SOURCE / "hubconf.py").is_file():
        raise FileNotFoundError(f"offline DINOv3 Torch Hub source is missing: {DINO_HUB_SOURCE}")
    if destination.exists() and any(destination.iterdir()):
        if not force:
            raise FileExistsError(f"bundle already exists: {destination}; rerun with --force")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    selection = yaml.safe_load((ROOT / "configs" / "regions" / "rp001.yaml").read_text())
    products = {row["product_id"]: row for row in selection["products"]}
    data_destination = destination / "data" / "rp001"
    records = []
    for product_id, product in sorted(products.items()):
        source_meta_path = REGION_ROOT / "metadata" / f"{product_id}.json"
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        raster = Path(source_meta["processed_science_raster"])
        if not raster.is_file():
            raise FileNotFoundError(f"processed raster is missing: {raster}")
        xml = Path(source_meta["raw_geotiff"]).with_suffix(".xml")
        metadata = read_lroc_metadata(raster, xml)
        raster_relative = Path("processed") / raster.name
        _copy(raster, data_destination / raster_relative)
        descriptor_relative = Path("reference_bank") / "descriptors" / f"{product_id}.npy"
        descriptor_destination = data_destination / descriptor_relative
        descriptor_destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(descriptor_destination, _descriptor(raster))
        thumbnail_relative = Path("reference_bank") / "thumbnails" / f"{product_id}.png"
        thumbnail = REGION_ROOT / "reference_bank" / "thumbnails" / f"{product_id}.png"
        _copy(thumbnail, data_destination / thumbnail_relative)
        records.append(
            {
                "product_id": product_id,
                "role": product["role"],
                "acquisition_date": product["acquisition_date"],
                "incidence_deg": float(product["incidence_deg"]),
                "emission_deg": float(product["emission_deg"]),
                "phase_deg": float(product["phase_deg"]),
                "gsd_m_per_px": float(product["gsd_m_per_px"]),
                "sha256": product["sha256"],
                "raster": str(raster_relative),
                "descriptor": str(descriptor_relative),
                "thumbnail": str(thumbnail_relative),
                "metadata": _metadata_payload(metadata, str(raster_relative)),
            }
        )
    (data_destination / "reference_bank" / "index.json").write_text(
        json.dumps(
            {
                "region_id": "RP-001",
                "method": "deterministic_normalized_intensity_histogram_cosine_ranking",
                "eligible_reference_roles": ["TRAIN"],
                "entry_count": 12,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_objects = {}
    for record in records:
        record_objects[record["product_id"]] = {
            "raster": REGION_ROOT / "processed" / Path(record["raster"]).name,
            "metadata": read_lroc_metadata(
                REGION_ROOT / "processed" / Path(record["raster"]).name,
                Path(
                    json.loads(
                        (REGION_ROOT / "metadata" / f"{record['product_id']}.json").read_text()
                    )["raw_geotiff"]
                ).with_suffix(".xml"),
            ),
        }
    parity_source = record_objects["NAC_PHO_E009S3481_M1138987738R"]
    parity_target = record_objects["NAC_PHO_E009S3481_M1096572724R"]
    _parity_fixture(
        data_destination / "fixtures" / "heldout_validation_parity.npz",
        parity_source,
        parity_target,
        (880, 4960),
    )
    negative_hash = _build_hard_negative(data_destination / "examples" / "other_lunar_region.npy")
    paired = json.loads(
        (ROOT / "runs" / "rp001_demo_adaptation_v1" / "paired_evaluation.json").read_text()
    )
    metrics = paired["comparison"]["TEST"]["official_romav2"]
    presentation_metrics = {
        "observations": {"display": "20", "exact": 20},
        "training_acquisitions": {"display": "12", "exact": 12},
        "valid_pairs": {"display": "19", "exact": 19},
        "dense_gt_crops": {"display": "97", "exact": 97},
        "test_vrr": {"display": "100% · 4/4", "exact": "4/4 = 1.0"},
        "test_pck_1": {"display": "82.6%", "exact": metrics["dense_gt"]["pck_1"]},
        "test_median_epe": {"display": "0.35 px", "exact": metrics["dense_gt"]["median_epe_px"]},
    }
    manifest = {
        "schema_version": 1,
        "region_id": "RP-001",
        "site_name": "Apollo 15 S-IVB Impact",
        "model": {"name": "Official RoMa v2", "sha256": sha256_file(CHECKPOINT)},
        "observations": records,
        "presentation_metrics": presentation_metrics,
        "scientific_details": {
            "official_test": metrics,
            "d1_status": "EXPERIMENTAL · NOT PROMOTED",
            "d1_selected": {"lr": "3e-6", "step": 10, "ema": True},
            "d1_reason": "D1 did not improve held-out validation performance.",
            "metric_protocol": (
                "paired base-640 protocol; exact metrics retained from paired_evaluation.json"
            ),
        },
        "known_cases": {
            "heldout": {
                "kind": "observation",
                "title": "KNOWN HELD-OUT",
                "query_product_id": "NAC_PHO_E009S3481_M1138987738R",
                "role": "VALIDATION",
            },
            "difficult_lighting": {
                "kind": "observation",
                "title": "DIFFICULT LIGHTING",
                "query_product_id": "NAC_PHO_E009S3481_M1177841115R",
                "role": "TEST",
            },
            "hard_negative": {
                "kind": "hard_negative",
                "title": "HARD NEGATIVE",
                "query_array": "examples/other_lunar_region.npy",
                "sha256": negative_hash,
                "reference_product_id": "NAC_PHO_E009S3481_M1096572724R",
                "reference_origin": [880, 4960],
            },
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (data_destination / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _copy(CHECKPOINT, destination / "models" / "romav2_official.pt")
    _copy_tree(
        DINO_HUB_SOURCE,
        destination / "models" / "torch_hub" / DINO_HUB_SOURCE.name,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "notebooks"),
    )
    for source, relative in (
        (ROOT / "api", Path("api")),
        (ROOT / "chandrappan", Path("chandrappan")),
        (ROOT / "frontend", Path("frontend")),
        (ROOT / "vendor" / "romav2" / "src", Path("vendor") / "romav2" / "src"),
    ):
        _copy_tree(
            source, destination / relative, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in (
        "demo_mac_m4.yaml",
        "regions/rp001.yaml",
    ):
        _copy(ROOT / "configs" / name, destination / "configs" / name)
    for name in (
        "requirements-demo-mac.txt",
        "pyproject.toml",
        "scripts/setup_mac_m4.sh",
        "scripts/start_demo_mac.sh",
        "scripts/doctor_mac_demo.py",
        "scripts/preflight_demo.sh",
        "scripts/benchmark_demo_resolution.py",
        "scripts/compare_demo_devices.py",
        "scripts/run_rp001_live_demo.py",
    ):
        _copy(ROOT / name, destination / name)
    static_source = ROOT / "demo" / "RP-001" / "final"
    static_destination = destination / "demo" / "RP-001" / "final"
    _copy_tree(static_source, static_destination, ignore=shutil.ignore_patterns("mac_inference"))
    _copy(ROOT / "frontend" / "static_fallback.html", static_destination / "index.html")
    (destination / "artifacts" / "live").mkdir(parents=True, exist_ok=True)
    transfer = _write_transfer_manifest(destination)
    (destination / "BUNDLE_README.md").write_text(
        "# Chandrappan Mac M4 demo\n\nRun `./scripts/setup_mac_m4.sh` once, then "
        "`./scripts/start_demo_mac.sh`. The presentation path is local-only and uses the "
        "frozen Official RoMa v2 checkpoint. D1 is not included in runtime loading.\n",
        encoding="utf-8",
    )
    return {**transfer, "manifest_sha256": manifest["manifest_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "release" / "chandrappan_mac_m4_demo")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output, force=args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
