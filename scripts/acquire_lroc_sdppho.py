#!/usr/bin/env python3
"""Acquire repeat-observation LROC SDPPHO GeoTIFF/XML pairs from official ODE records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.lroc_ode import (  # noqa: E402
    ODE_SDPPHO_QUERY,
    discover_sites,
    download_product_files,
    parse_ode_products,
    select_acquisition_products,
)


def _load_or_fetch(path: Path, limit: int | None) -> bytes:
    if path.exists() and path.stat().st_size > 1024:
        return path.read_bytes()
    url = ODE_SDPPHO_QUERY + (f"&limit={limit}" if limit else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        payload = response.read()
    path.write_bytes(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ode-cache", type=Path, default=Path("data/raw/lroc_index/ode_sdppho_products_1000.json")
    )
    parser.add_argument("--ode-limit", type=int, default=1000)
    parser.add_argument("--destination", type=Path, default=Path("data/raw/lroc_corpus"))
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--products-per-region", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--site-index", type=Path, default=Path("results/lroc_site_discovery.json"))
    parser.add_argument(
        "--provenance", type=Path, default=Path("data/raw/lroc_corpus/provenance.json")
    )
    args = parser.parse_args()

    products = parse_ode_products(_load_or_fetch(args.ode_cache, args.ode_limit))
    sites = discover_sites(products)
    args.site_index.parent.mkdir(parents=True, exist_ok=True)
    args.site_index.write_text(
        json.dumps([site.as_dict() for site in sites], indent=2) + "\n",
        encoding="utf-8",
    )
    selected = select_acquisition_products(
        products,
        regions=[region.upper() for region in args.regions],
        products_per_region=args.products_per_region,
    )
    if not selected:
        raise RuntimeError("no products matched requested regions")
    new_rows = download_product_files(selected, args.destination, overwrite=args.overwrite)
    old_rows = []
    if args.provenance.exists():
        old_rows = json.loads(args.provenance.read_text(encoding="utf-8"))
    dedup = {(row["path"], row["kind"]): row for row in old_rows + new_rows}
    args.provenance.write_text(
        json.dumps(list(dedup.values()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "regions": sorted({row.region_id for row in selected}),
                "products": len(selected),
                "files_written_or_verified": len(new_rows),
                "site_index": str(args.site_index),
                "provenance": str(args.provenance),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
