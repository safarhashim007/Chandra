#!/usr/bin/env python3
"""Acquire and verify immutable NASA files for a frozen Region Pack selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.data.lroc_ode import parse_ode_products
from chandrappan.data.pds3 import read_pds3_header


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_atomic(url: str, destination: Path, *, overwrite: bool = False) -> None:
    """Download to a sibling part file, retaining an already verified file by default."""
    if destination.exists() and not overwrite:
        if destination.stat().st_size == 0:
            raise ValueError(f"refusing zero-byte existing file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Chandrappan-RP-001/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if partial.stat().st_size == 0:
            raise ValueError(f"server returned a zero-byte response for {url}")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _file_url(product: Any, filename: str) -> str:
    files = product.__dict__
    mapping = {
        f"{product.product_id}.IMG": files["product_img_url"],
        f"{product.product_id}.TIF": files["geotiff_url"],
        f"{product.product_id}.XML": files["label_url"],
    }
    value = mapping.get(filename.upper())
    if not value:
        raise ValueError(f"official ODE record lacks {filename} for {product.product_id}")
    return str(value)


def acquire(selection: Path, ode_cache: Path, *, overwrite: bool = False) -> dict[str, object]:
    config = yaml.safe_load(selection.read_text(encoding="utf-8"))
    if config.get("region_id") != "RP-001":
        raise ValueError("this acquisition command only accepts the RP-001 selection")
    selected = config.get("products") or []
    if len(selected) < 16:
        raise ValueError("RP-001 selection has fewer than 16 observations")
    if len({row["product_id"] for row in selected}) != len(selected):
        raise ValueError("RP-001 selection contains duplicate product IDs")

    products = {item.product_id: item for item in parse_ode_products(ode_cache.read_bytes())}
    rows: list[dict[str, object]] = []
    for item in selected:
        product_id = item["product_id"]
        try:
            product = products[product_id]
        except KeyError as exc:
            raise ValueError(
                f"selected product missing from the official ODE cache: {product_id}"
            ) from exc
        if product.product_img_url != item["source_url"]:
            raise ValueError(f"source URL drift for {product_id}; re-audit before downloading")
        raw_img = Path(item["local_path"])
        raw_tif = raw_img.with_suffix(".TIF")
        raw_xml = raw_img.with_suffix(".xml")
        for url, destination in (
            (product.product_img_url, raw_img),
            (product.geotiff_url, raw_tif),
            (product.label_url, raw_xml),
        ):
            if url is None:
                raise ValueError(f"official ODE record lacks a required URL for {product_id}")
            download_atomic(str(url), destination, overwrite=overwrite)

        header = read_pds3_header(raw_img)
        if header.product_id != product_id:
            raise ValueError(f"PDS3 product ID mismatch in {raw_img}: {header.product_id}")
        metadata = read_lroc_metadata(raw_tif, raw_xml)
        if (metadata.width, metadata.height) != (header.line_samples, header.lines):
            raise ValueError(
                f"browse raster dimensions disagree with original PDS3 image for {product_id}"
            )
        rows.append(
            {
                "product_id": product_id,
                "role": item["role"],
                "source": "NASA_LROC_ODE_PDS",
                "pds_product_lid": product.product_lid,
                "raw_img": {
                    "path": str(raw_img),
                    "url": product.product_img_url,
                    "bytes": raw_img.stat().st_size,
                    "sha256": sha256_file(raw_img),
                },
                "raw_geotiff": {
                    "path": str(raw_tif),
                    "url": product.geotiff_url,
                    "bytes": raw_tif.stat().st_size,
                    "sha256": sha256_file(raw_tif),
                },
                "label": {
                    "path": str(raw_xml),
                    "url": product.label_url,
                    "bytes": raw_xml.stat().st_size,
                    "sha256": sha256_file(raw_xml),
                },
                "validation": {
                    "pds3_header": header.as_dict(),
                    "browse_geotiff_product_id": metadata.product_id,
                    "browse_geotiff_dimensions": [metadata.width, metadata.height],
                    "browse_geotiff_gsd_m_per_px": metadata.gsd_m_per_px,
                },
            }
        )
    root = Path(selected[0]["local_path"]).parents[1]
    provenance = root / "metadata" / "provenance.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "region_id": config["region_id"],
        "selection_config": str(selection),
        "source_cache": str(ode_cache),
        "product_count": len(rows),
        "products": rows,
    }
    provenance.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"provenance": str(provenance), "product_count": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("configs/regions/rp001.yaml"))
    parser.add_argument(
        "--ode-cache", type=Path, default=Path("data/raw/lroc_index/ode_sdppho_products_1000.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(acquire(args.selection, args.ode_cache, overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
