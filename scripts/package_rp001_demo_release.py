#!/usr/bin/env python3
"""Build the self-contained, transfer-ready RP-001 demo image package."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
PACKAGE = RELEASE / "CHANDRAPPAN_DEMO_IMAGES"

CASES = [
    {
        "case": "01_PRIMARY_VERIFIED",
        "role": "PRIMARY",
        "source": ROOT
        / "data/region_packs/RP-001/processed/NAC_PHO_E009S3481_M1345996066R_science.tif",
        "product_id": "NAC_PHO_E009S3481_M1345996066R",
        "split": "TEST",
        "expected_verdict": "VERIFIED",
        "reference": ROOT
        / "data/region_packs/RP-001/processed/NAC_PHO_E009S3481_M1118958225R_science.tif",
        "reference_product_id": "NAC_PHO_E009S3481_M1118958225R",
        "reference_rank": 1,
        "source_region": "RP-001",
        "source_url": "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_PHO/E009S3481/NAC_PHO_E009S3481_M1345996066R.IMG",
    },
    {
        "case": "02_DIFFICULT_LIGHTING_VERIFIED",
        "role": "DIFFICULT_LIGHTING",
        "source": ROOT
        / "data/region_packs/RP-001/processed/NAC_PHO_E009S3481_M1177841115R_science.tif",
        "product_id": "NAC_PHO_E009S3481_M1177841115R",
        "split": "TEST",
        "expected_verdict": "VERIFIED",
        "reference": ROOT
        / "data/region_packs/RP-001/processed/NAC_PHO_E009S3481_M106949300R_science.tif",
        "reference_product_id": "NAC_PHO_E009S3481_M106949300R",
        "reference_rank": 3,
        "source_region": "RP-001",
        "source_url": "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_PHO/E009S3481/NAC_PHO_E009S3481_M1177841115R.IMG",
    },
    {
        "case": "03_FAILED_HELDOUT_CASE",
        "role": "FAILED_HELDOUT_CASE",
        "source": ROOT
        / "data/region_packs/RP-001/processed/NAC_PHO_E009S3481_M177711422R_science.tif",
        "product_id": "NAC_PHO_E009S3481_M177711422R",
        "split": "TEST",
        "expected_verdict": "REJECTED",
        "reference": ROOT
        / "data/region_packs/RP-001/processed/NAC_PHO_E009S3481_M160030722R_science.tif",
        "reference_product_id": "NAC_PHO_E009S3481_M160030722R",
        "reference_rank": 1,
        "source_region": "RP-001",
        "source_url": "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_PHO/E009S3481/NAC_PHO_E009S3481_M177711422R.IMG",
    },
    {
        "case": "04_HARD_NEGATIVE",
        "role": "HARD_NEGATIVE",
        "source": ROOT / "data/raw/lroc_corpus/NAC_PHO_E018N3346_M1142603254L.TIF",
        "product_id": "NAC_PHO_E018N3346_M1142603254L",
        "split": "OUTSIDE_RP-001",
        "expected_verdict": "REJECTED",
        "reference": ROOT
        / "data/region_packs/RP-001/processed/NAC_PHO_E009S3481_M160030722R_science.tif",
        "reference_product_id": "NAC_PHO_E009S3481_M160030722R",
        "reference_rank": None,
        "source_region": "E018N3346",
        "source_url": "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/BROWSE/NAC_PHO/E018N3346/NAC_PHO_E018N3346_M1142603254L.TIF",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raster_info(path: Path) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        return {
            "width": dataset.width,
            "height": dataset.height,
            "format": "TIFF"
            if path.suffix.lower() in {".tif", ".tiff"}
            else path.suffix.lower().lstrip("."),
            "dtype": dataset.dtypes[0],
            "bands": dataset.count,
        }


def exact_png(source: Path, destination: Path) -> dict[str, object]:
    """Write a same-size, same-pixel grayscale PNG for uint8 science rasters."""
    with rasterio.open(source) as dataset:
        values = dataset.read(1)
    if values.dtype != np.uint8:
        raise ValueError(f"expected uint8 source for exact PNG copy: {source} ({values.dtype})")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), values, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError(f"failed to write {destination}")
    return {
        "conversion": (
            "TIFF uint8 band 1 -> grayscale PNG, same width/height, same pixel values, "
            "no crop/rotation/flip/enhancement"
        ),
        "source_sha256": sha256(source),
    }


def display_preview(source: Path, destination: Path, label: str) -> None:
    with rasterio.open(source) as dataset:
        values = dataset.read(
            1, out_shape=(1, 720, 960), resampling=rasterio.enums.Resampling.average
        ).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0)
    low, high = np.percentile(values, (1, 99))
    image = np.clip((values - low) / max(high - low, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        image, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def copy_evidence(case: dict[str, object], destination: Path) -> None:
    """Copy previously generated official-result artifacts into the case folder."""
    old_role = {
        "01_PRIMARY_VERIFIED": "primary",
        "02_DIFFICULT_LIGHTING_VERIFIED": "difficult_lighting",
        "03_FAILED_HELDOUT_CASE": "backup",
        "04_HARD_NEGATIVE": "hard_negative",
    }[str(case["case"])]
    existing = ROOT / "demo/RP-001/handpicked" / old_role
    for name in ("matches.png", "overlay.png", "difference.png", "presentation_card.png"):
        source = existing / name
        if source.exists():
            shutil.copy2(source, destination / name)
    # Wipe-compatible assets are explicit aliases/copies, never symlinks.
    if (destination / "overlay.png").exists():
        shutil.copy2(destination / "overlay.png", destination / "registered_overlay.png")
    if (destination / "difference.png").exists():
        shutil.copy2(destination / "difference.png", destination / "registered_difference.png")
    result = existing / "result.json"
    if result.exists():
        payload = json.loads(result.read_text(encoding="utf-8"))
    else:
        payload = {"demo_id": case["case"], "verdict": case["expected_verdict"]}
    payload["release_package_case"] = case["case"]
    payload["expected_verdict"] = case["expected_verdict"]
    (destination / "result.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    for name in (
        "UPLOAD_THESE",
        "PREVIEWS",
        "REFERENCES",
        "01_PRIMARY_VERIFIED",
        "02_DIFFICULT_LIGHTING_VERIFIED",
        "03_FAILED_HELDOUT_CASE",
        "04_HARD_NEGATIVE",
    ):
        (PACKAGE / name).mkdir(parents=True)

    records: list[dict[str, object]] = []
    important_paths: list[Path] = []
    for case in CASES:
        case_name = str(case["case"])
        source = Path(case["source"])
        reference = Path(case["reference"])
        case_dir = PACKAGE / case_name
        if not source.is_file() or not reference.is_file():
            raise FileNotFoundError(f"missing source/reference: {source} / {reference}")
        query_tif = PACKAGE / "UPLOAD_THESE" / f"{case_name}.tif"
        query_png = PACKAGE / "UPLOAD_THESE" / f"{case_name}.png"
        shutil.copy2(source, query_tif)
        png_meta = exact_png(source, query_png)
        ref_tif = PACKAGE / "REFERENCES" / f"{case_name}_selected_reference.tif"
        shutil.copy2(reference, ref_tif)
        ref_png = PACKAGE / "REFERENCES" / f"{case_name}_selected_reference.png"
        exact_png(reference, ref_png)
        display_preview(source, PACKAGE / "PREVIEWS" / f"{case_name}_preview.png", case_name)

        # Case-local copies contain actual image data and evidence, not links.
        local_query = case_dir / "query_original.tif"
        local_png = case_dir / "query.png"
        local_ref = case_dir / "selected_reference_original.tif"
        local_ref_png = case_dir / "selected_reference.png"
        shutil.copy2(source, local_query)
        shutil.copy2(query_png, local_png)
        shutil.copy2(reference, local_ref)
        shutil.copy2(ref_png, local_ref_png)
        copy_evidence(case, case_dir)

        source_info = raster_info(source)
        query_record = {
            "demo_case": case_name,
            "role": case["role"],
            "filename": query_tif.name,
            "original_source_path": str(source.relative_to(ROOT)),
            "copied_path": str(query_tif.relative_to(PACKAGE)),
            "product_id": case["product_id"],
            "region": case["source_region"],
            "split": case["split"],
            "expected_verdict": case["expected_verdict"],
            "selected_reference": case["reference_product_id"],
            "selected_reference_rank": case["reference_rank"],
            "used_for_training": False,
            "sha256": sha256(query_tif),
            "original_sha256": sha256(source),
            "byte_for_byte_copy": sha256(source) == sha256(query_tif),
            "file_size_bytes": query_tif.stat().st_size,
            **source_info,
            "png_convenience_copy": {
                "filename": query_png.name,
                "path": str(query_png.relative_to(PACKAGE)),
                "sha256": sha256(query_png),
                "file_size_bytes": query_png.stat().st_size,
                **png_meta,
            },
            "source_url": case["source_url"],
        }
        records.append(query_record)
        important_paths.extend([query_tif, query_png, ref_tif, ref_png])

        provenance = {
            "demo_case": case_name,
            "query_product_id": case["product_id"],
            "query_source_path": str(source.relative_to(ROOT)),
            "query_source_sha256": sha256(source),
            "query_copied_sha256": sha256(query_tif),
            "query_split": case["split"],
            "expected_verdict": case["expected_verdict"],
            "selected_reference_product_id": case["reference_product_id"],
            "selected_reference_split": "TRAIN",
            "selected_reference_rank": case["reference_rank"],
            "reference_source_path": str(reference.relative_to(ROOT)),
            "reference_sha256": sha256(reference),
            "source_region": case["source_region"],
            "matcher": "official RoMa v2, precise bidirectional setting",
            "used_for_training": False,
            "conversion": png_meta["conversion"],
        }
        (case_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )

    # Correctly named reference aliases requested by the presentation bundle spec.
    shutil.copy2(
        PACKAGE / "REFERENCES/01_PRIMARY_VERIFIED_selected_reference.tif",
        PACKAGE / "REFERENCES/01_PRIMARY_selected_reference.tif",
    )
    shutil.copy2(
        PACKAGE / "REFERENCES/01_PRIMARY_VERIFIED_selected_reference.png",
        PACKAGE / "REFERENCES/01_PRIMARY_selected_reference.png",
    )
    shutil.copy2(
        PACKAGE / "REFERENCES/02_DIFFICULT_LIGHTING_VERIFIED_selected_reference.tif",
        PACKAGE / "REFERENCES/02_DIFFICULT_LIGHTING_selected_reference.tif",
    )
    shutil.copy2(
        PACKAGE / "REFERENCES/02_DIFFICULT_LIGHTING_VERIFIED_selected_reference.png",
        PACKAGE / "REFERENCES/02_DIFFICULT_LIGHTING_selected_reference.png",
    )
    shutil.copy2(
        PACKAGE / "REFERENCES/03_FAILED_HELDOUT_CASE_selected_reference.tif",
        PACKAGE / "REFERENCES/03_FAILED_highest_ranked_reference.tif",
    )
    shutil.copy2(
        PACKAGE / "REFERENCES/04_HARD_NEGATIVE_selected_reference.tif",
        PACKAGE / "REFERENCES/04_NEGATIVE_attempted_reference.tif",
    )

    # README is deliberately plain for a presenter.
    readme = """# CHANDRAPPAN DEMO IMAGES

## USE THESE DURING THE PRESENTATION

Open:

`UPLOAD_THESE/`

### 1. PRIMARY

Upload:

`01_PRIMARY_VERIFIED.tif`

Expected:

`VERIFIED`

This is the safest main demonstration.

### 2. DIFFICULT LIGHTING

Upload:

`02_DIFFICULT_LIGHTING_VERIFIED.tif`

Expected:

`VERIFIED`

Use this to demonstrate changed lunar illumination.

### 3. FAILED HELD-OUT CASE

Upload:

`03_FAILED_HELDOUT_CASE.tif`

Expected:

`REJECTED`

Do NOT use this as the first demo. Use only if explaining limitations.

### 4. HARD NEGATIVE

Upload:

`04_HARD_NEGATIVE.tif`

Expected:

`REJECTED`

Use this after the successful demos to show geometric false-match rejection.

## RECOMMENDED PRESENTATION ORDER

1. `01_PRIMARY_VERIFIED`
2. `02_DIFFICULT_LIGHTING_VERIFIED`
3. `04_HARD_NEGATIVE`

Do not normally show case 03 unless asked about failures.

The TIFF files are the preferred inference inputs. The PNG files are deterministic,
same-size, same-pixel-value convenience copies for Finder/browser upload.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "package": "CHANDRAPPAN_DEMO_IMAGES",
        "status": "FAIL_AS_REQUESTED_THIRD_VERIFIED_BACKUP_UNAVAILABLE",
        "matcher": "official RoMa v2, precise bidirectional setting",
        "cases": records,
        "reference_files": [],
        "files": [],
    }
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file():
            rel = path.relative_to(PACKAGE)
            parts = rel.parts
            case_name = parts[0] if parts and parts[0] in {str(c["case"]) for c in CASES} else None
            if (
                case_name is None
                and len(parts) > 1
                and parts[0] in {"UPLOAD_THESE", "PREVIEWS", "REFERENCES"}
            ):
                stem = path.stem
                case_name = next(
                    (str(c["case"]) for c in CASES if stem.startswith(str(c["case"]))), None
                )
            case_info = next((c for c in CASES if str(c["case"]) == case_name), None)
            info = {
                "path": str(rel),
                "filename": path.name,
                "demo_case": case_name,
                "original_source_path": str(case_info["source"].relative_to(ROOT))
                if case_info
                else None,
                "expected_verdict": case_info["expected_verdict"] if case_info else None,
                "sha256": sha256(path),
                "file_size_bytes": path.stat().st_size,
                "format": path.suffix.lower().lstrip(".") or "text",
            }
            if path.suffix.lower() in {".tif", ".tiff"}:
                info.update(raster_info(path))
            manifest["files"].append(info)
    manifest["reference_files"] = [
        x for x in manifest["files"] if str(x["path"]).startswith("REFERENCES/")
    ]
    (PACKAGE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Hash every package file, including metadata and visual evidence.
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(PACKAGE)}"
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    (PACKAGE / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    # Copy the complete folder into the Mac release bundle.
    mac_bundle = RELEASE / "chandrappan_mac_m4_demo/demo_images"
    if mac_bundle.exists():
        shutil.rmtree(mac_bundle)
    shutil.copytree(PACKAGE, mac_bundle)

    zip_path = RELEASE / "CHANDRAPPAN_DEMO_IMAGES.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                archive.write(path, Path(PACKAGE.name) / path.relative_to(PACKAGE))

    required = [
        PACKAGE / "UPLOAD_THESE/01_PRIMARY_VERIFIED.tif",
        PACKAGE / "UPLOAD_THESE/02_DIFFICULT_LIGHTING_VERIFIED.tif",
        PACKAGE / "UPLOAD_THESE/03_FAILED_HELDOUT_CASE.tif",
        PACKAGE / "UPLOAD_THESE/04_HARD_NEGATIVE.tif",
        PACKAGE / "README.md",
        PACKAGE / "manifest.json",
        PACKAGE / "SHA256SUMS.txt",
        zip_path,
    ]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError("package verification failed: " + ", ".join(missing))
    print(f"PACKAGE={PACKAGE}")
    print(f"ZIP={zip_path}")
    print("MISSING=none")
    print("DEMO IMAGE TRANSFER PACKAGE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
