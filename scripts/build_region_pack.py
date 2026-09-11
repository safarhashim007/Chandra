#!/usr/bin/env python3
"""Build the non-model artifacts for an acquired, frozen lunar Region Pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.gt_warp import dense_pixel_warp
from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.data.pds3 import read_pds3_header, sampled_band_statistics
from chandrappan.geo.footprint import footprint_corners_world
from chandrappan.region_packs import intensity_descriptor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_acquisition_split(products: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Reject any observation-level split leakage before building a pack."""
    groups = {"TRAIN": [], "VALIDATION": [], "TEST": []}
    seen: set[str] = set()
    for product in products:
        product_id = str(product["product_id"])
        role = product.get("role")
        if role not in groups:
            raise ValueError(f"unsupported RP-001 role for {product_id}: {role}")
        if product_id in seen:
            raise ValueError(f"acquisition leakage: {product_id} appears more than once")
        seen.add(product_id)
        groups[role].append(product_id)
    if len(products) < 16:
        raise ValueError("Region Pack needs at least 16 acquisition-isolated observations")
    if any(not value for value in groups.values()):
        raise ValueError("TRAIN, VALIDATION, and TEST must each contain whole acquisitions")
    if len(products) >= 20 and not (
        0.5 <= len(groups["TRAIN"]) / len(products) <= 0.7
        and 0.15 <= len(groups["VALIDATION"]) / len(products) <= 0.25
        and 0.15 <= len(groups["TEST"]) / len(products) <= 0.25
    ):
        raise ValueError("20+ observation split must remain approximately 60/20/20")
    return groups


def _crop_metadata(metadata: Any, center_world: tuple[float, float], size: int) -> Any:
    center_px = metadata.transform.world_to_pixel(*center_world)
    origin_x = max(0, min(metadata.width - size, int(center_px[0] - size / 2)))
    origin_y = max(0, min(metadata.height - size, int(center_px[1] - size / 2)))
    world_origin = metadata.transform.pixel_to_world(origin_x, origin_y)
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


def _overlap_center(first: Any, second: Any) -> tuple[float, float]:
    overlap = Polygon(footprint_corners_world(first)).intersection(
        Polygon(footprint_corners_world(second))
    )
    if overlap.is_empty:
        raise ValueError(
            f"no physical map overlap between {first.product_id} and {second.product_id}"
        )
    return float(overlap.centroid.x), float(overlap.centroid.y)


def _cycle_error(source: Any, target: Any, warp: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    target_x, target_y = warp[..., 0], warp[..., 1]
    world_x, world_y = target.transform.pixel_to_world(target_x, target_y)
    source_x, source_y = source.transform.world_to_pixel(world_x, world_y)
    y, x = np.indices(valid.shape, dtype=np.float64)
    error = np.hypot(source_x - (x + 0.5), source_y - (y + 0.5))[valid]
    if error.size == 0:
        raise ValueError("dense GT has no valid cycle points")
    return {"median_px": float(np.median(error)), "max_px": float(np.max(error))}


def _write_thumbnail(source: Path, destination: Path, annotation: str) -> None:
    import cv2
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(source) as dataset:
        values = dataset.read(1, out_shape=(1, 180, 240), resampling=Resampling.average)
    values = np.nan_to_num(values.astype(np.float32), nan=0.0)
    low, high = np.percentile(values, (1, 99))
    image = np.clip((values - low) * 255 / (high - low + 1e-6), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, annotation, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise RuntimeError(f"failed to write thumbnail {destination}")


def _write_contact_sheet(items: list[dict[str, Any]], destination: Path, title: str) -> None:
    import cv2

    thumbnails = []
    for item in items:
        image = cv2.imread(str(item["thumbnail"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"missing thumbnail for contact sheet: {item['thumbnail']}")
        canvas = np.zeros((220, 240, 3), dtype=np.uint8)
        canvas[:180] = image
        cv2.putText(
            canvas,
            item["product_id"].replace("NAC_PHO_E009S3481_", ""),
            (5, 197),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            f"I {item['incidence_deg']:.1f} E {item['emission_deg']:.1f} P {item['phase_deg']:.1f}",
            (5, 214),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
        )
        thumbnails.append(canvas)
    columns = 4
    rows = [thumbnails[index : index + columns] for index in range(0, len(thumbnails), columns)]
    blank = np.zeros_like(thumbnails[0])
    panel_rows = [cv2.hconcat(row + [blank] * (columns - len(row))) for row in rows]
    panel = cv2.vconcat(panel_rows)
    header = np.zeros((35, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.vconcat((header, panel))):
        raise RuntimeError(f"failed to write contact sheet {destination}")


def _build_gt(
    observations: list[dict[str, Any]], root: Path, *, crop_size: int
) -> dict[str, object]:
    by_role = {
        role: [row for row in observations if row["role"] == role]
        for role in ("TRAIN", "VALIDATION", "TEST")
    }
    requested: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    requested.extend(
        (first, second, "train") for first, second in combinations(by_role["TRAIN"], 2)
    )
    requested.extend(
        (query, reference, "validation_reference")
        for query in by_role["VALIDATION"]
        for reference in by_role["TRAIN"]
    )
    requested.extend(
        (query, reference, "test_reference")
        for query in by_role["TEST"]
        for reference in by_role["TRAIN"]
    )
    index = []
    gt_root = root / "gt"
    gt_root.mkdir(parents=True, exist_ok=True)
    for first, second, purpose in requested:
        metadata_a, metadata_b = first["metadata_obj"], second["metadata_obj"]
        center = _overlap_center(metadata_a, metadata_b)
        crop_a = _crop_metadata(metadata_a, center, crop_size)
        crop_b = _crop_metadata(metadata_b, center, crop_size)
        warp_ab, valid_ab = dense_pixel_warp(crop_a, crop_b)
        warp_ba, valid_ba = dense_pixel_warp(crop_b, crop_a)
        cycle_ab = _cycle_error(crop_a, crop_b, warp_ab, valid_ab)
        cycle_ba = _cycle_error(crop_b, crop_a, warp_ba, valid_ba)
        if cycle_ab["max_px"] >= 0.01 or cycle_ba["max_px"] >= 0.01:
            raise ValueError(
                "geospatial GT cycle exceeds 0.01 px for "
                f"{first['product_id']} / {second['product_id']}"
            )
        pair_id = f"{first['product_id']}__{second['product_id']}"
        output = gt_root / f"{pair_id}.npz"
        np.savez_compressed(
            output,
            warp_ab_px=warp_ab,
            warp_ba_px=warp_ba,
            valid_ab=valid_ab,
            valid_ba=valid_ba,
            overlap_ab=valid_ab,
            overlap_ba=valid_ba,
        )
        index.append(
            {
                "pair_id": pair_id,
                "purpose": purpose,
                "path": str(output),
                "crop_size_px": crop_size,
                "overlap_center_world_m": [center[0], center[1]],
                "valid_ab_coverage": float(valid_ab.mean()),
                "valid_ba_coverage": float(valid_ba.mean()),
                "cycle_ab": cycle_ab,
                "cycle_ba": cycle_ba,
                "supervision_note": (
                    "geospatial/map-derived correspondence, not surveyed landmark truth"
                ),
            }
        )
    (gt_root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return {"pair_count": len(index), "index": str(gt_root / "index.json")}


def build(selection_path: Path, *, crop_size: int = 640) -> dict[str, object]:
    config = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    if config.get("region_id") != "RP-001":
        raise ValueError("expected the RP-001 configuration")
    products = config["products"]
    split = validate_acquisition_split(products)
    root = Path(products[0]["local_path"]).parents[1]
    processed = root / "processed"
    metadata_root = root / "metadata"
    bank_root = root / "reference_bank"
    observations = []
    canonical_footprint = None
    for product in products:
        raw_img = Path(product["local_path"])
        raw_tif = raw_img.with_suffix(".TIF")
        raw_xml = raw_img.with_suffix(".xml")
        header = read_pds3_header(raw_img)
        if header.product_id != product["product_id"]:
            raise ValueError(f"raw product mismatch for {raw_img}")
        if header.start_time != product["acquisition_date"]:
            raise ValueError(f"acquisition timestamp drift for {header.product_id}")
        metadata = read_lroc_metadata(raw_tif, raw_xml)
        stats = sampled_band_statistics(raw_img, header)
        science_tif = processed / f"{header.product_id}_science.tif"
        science_tif.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_tif, science_tif)
        if sha256_file(raw_tif) != sha256_file(science_tif):
            raise ValueError(f"science raster copy changed original pixels for {header.product_id}")
        footprint = Polygon(footprint_corners_world(metadata))
        if canonical_footprint is None:
            canonical_footprint = footprint
        overlap = footprint.intersection(canonical_footprint).area / canonical_footprint.area
        product["sha256"] = sha256_file(raw_img)
        product["incidence_deg"] = stats["incidence_deg"]["median"]
        product["emission_deg"] = stats["emission_deg"]["median"]
        product["phase_deg"] = stats["phase_deg"]["median"]
        thumbnail = bank_root / "thumbnails" / f"{header.product_id}.png"
        _write_thumbnail(science_tif, thumbnail, product["role"])
        observation = {
            "product_id": header.product_id,
            "role": product["role"],
            "acquisition_date": header.start_time,
            "incidence_deg": product["incidence_deg"],
            "emission_deg": product["emission_deg"],
            "phase_deg": product["phase_deg"],
            "gsd_m_per_px": metadata.gsd_m_per_px,
            "width": metadata.width,
            "height": metadata.height,
            "center_lat": metadata.center_lat,
            "center_lon": metadata.center_lon_east,
            "footprint_area": float(footprint.area),
            "overlap_with_canonical": float(overlap),
            "quality_status": "VALIDATED",
            "raw_img": str(raw_img),
            "raw_geotiff": str(raw_tif),
            "processed_science_raster": str(science_tif),
            "thumbnail": str(thumbnail),
            "sha256": product["sha256"],
            "angle_statistics": stats,
            "metadata_obj": metadata,
        }
        metadata_root.mkdir(parents=True, exist_ok=True)
        (metadata_root / f"{header.product_id}.json").write_text(
            json.dumps(
                {
                    **{key: value for key, value in observation.items() if key != "metadata_obj"},
                    "pds3_header": header.as_dict(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        observations.append(observation)
    config["status"] = "ACQUIRED_VALIDATED_PENDING_EVALUATION"
    selection_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    catalog = root / "observation_catalog.csv"
    columns = [
        "product_id",
        "acquisition_date",
        "incidence_deg",
        "emission_deg",
        "phase_deg",
        "gsd_m_per_px",
        "width",
        "height",
        "center_lat",
        "center_lon",
        "footprint_area",
        "overlap_with_canonical",
        "role",
        "quality_status",
    ]
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: observation[key] for key in columns} for observation in observations)
    split_path = root / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "region_id": "RP-001",
                "isolation_unit": "full_observation_acquisition",
                "frozen": True,
                "splits": split,
                "leakage_check": "PASS",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    bank_entries = []
    import rasterio

    for observation in observations:
        if observation["role"] != "TRAIN":
            continue
        with rasterio.open(observation["processed_science_raster"]) as dataset:
            image = dataset.read(1, out_shape=(1, 256, 256))
        descriptor = intensity_descriptor(image)
        descriptor_path = bank_root / "descriptors" / f"{observation['product_id']}.npy"
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(descriptor_path, descriptor)
        bank_entries.append(
            {
                "product_id": observation["product_id"],
                "role": "TRAIN",
                "descriptor": str(descriptor_path),
                "thumbnail": observation["thumbnail"],
                "footprint_area_m2": observation["footprint_area"],
                "source_product_id": observation["product_id"],
            }
        )
    (bank_root / "index.json").write_text(
        json.dumps(
            {
                "region_id": "RP-001",
                "method": "deterministic_normalized_intensity_histogram_cosine_ranking",
                "candidate_only": True,
                "eligible_reference_roles": ["TRAIN"],
                "entries": bank_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for role in ("TRAIN", "VALIDATION", "TEST"):
        _write_contact_sheet(
            [row for row in observations if row["role"] == role],
            Path("demo/RP-001") / f"{role.lower()}_contact_sheet.png",
            f"RP-001 {role} — Apollo 15 S-IVB Impact",
        )
    gt = _build_gt(observations, root, crop_size=crop_size)

    def numeric(key: str) -> list[float]:
        return [float(row[key]) for row in observations]

    report = {
        "region_id": "RP-001",
        "validated_observations": len(observations),
        "role_counts": {role: len(values) for role, values in split.items()},
        "incidence_deg": {
            "min": min(numeric("incidence_deg")),
            "max": max(numeric("incidence_deg")),
        },
        "emission_deg": {"min": min(numeric("emission_deg")), "max": max(numeric("emission_deg"))},
        "phase_deg": {"min": min(numeric("phase_deg")), "max": max(numeric("phase_deg"))},
        "gsd_m_per_px": {"min": min(numeric("gsd_m_per_px")), "max": max(numeric("gsd_m_per_px"))},
        "acquisition_date_range": [
            min(row["acquisition_date"] for row in observations),
            max(row["acquisition_date"] for row in observations),
        ],
        "overlap_with_canonical": {
            "min": min(numeric("overlap_with_canonical")),
            "max": max(numeric("overlap_with_canonical")),
        },
        "dense_gt": gt,
        "reference_bank_entries": len(bank_entries),
    }
    diversity_md = Path("results/RP-001_observation_diversity.md")
    incidence, emission, phase = (
        report["incidence_deg"],
        report["emission_deg"],
        report["phase_deg"],
    )
    gsd, overlap = report["gsd_m_per_px"], report["overlap_with_canonical"]
    diversity_lines = [
        "# RP-001 observation diversity",
        "",
        f"- Validated observations: {report['validated_observations']}",
        (
            f"- Split: TRAIN {len(split['TRAIN'])}, VALIDATION {len(split['VALIDATION'])}, "
            f"TEST {len(split['TEST'])}"
        ),
        f"- Incidence (backplane median) range: {incidence['min']:.3f}–{incidence['max']:.3f}°",
        f"- Emission (backplane median) range: {emission['min']:.3f}–{emission['max']:.3f}°",
        f"- Phase (backplane median) range: {phase['min']:.3f}–{phase['max']:.3f}°",
        f"- GSD range: {gsd['min']:.3f}–{gsd['max']:.3f} m/px",
        f"- Canonical footprint overlap: {overlap['min']:.3%}–{overlap['max']:.3%}",
        (
            f"- Acquisition range: {report['acquisition_date_range'][0]} to "
            f"{report['acquisition_date_range'][1]}"
        ),
        "",
        "Angles are sampled statistics from the immutable official PDS3 backplanes. They are not "
        "assumed to be a single global observation angle. Dense GT is map-derived geospatial "
        "correspondence supervision, not surveyed physical landmark truth.",
        "",
    ]
    diversity_md.write_text("\n".join(diversity_lines), encoding="utf-8")
    return {"catalog": str(catalog), "split": str(split_path), "report": report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("configs/regions/rp001.yaml"))
    parser.add_argument("--crop-size", type=int, default=640)
    args = parser.parse_args()
    print(json.dumps(build(args.selection, crop_size=args.crop_size), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
