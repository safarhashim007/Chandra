#!/usr/bin/env python3
"""Build a real LROC NAC manifest and leakage-safe evaluation splits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.manifest import image_record_from_lroc, write_manifest
from chandrappan.data.pairs import generate_negative_pairs, generate_pairs, write_pairs
from chandrappan.data.splits import geographic_split, validate_no_leakage, write_split_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/expanded"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--preserve-splits", type=Path)
    parser.add_argument(
        "--provenance", type=Path, default=Path("data/raw/lroc_corpus/provenance.json")
    )
    parser.add_argument("rasters", nargs="*", type=Path)
    args = parser.parse_args()
    rasters = args.rasters or sorted(
        list(Path("data/raw/lroc_corpus").glob("*.TIF"))
        + list(Path("data/raw/lroc_pair").glob("*.TIF"))
    )
    if not rasters:
        raise RuntimeError("no LROC GeoTIFFs found")
    provenance = {}
    if args.provenance.exists():
        for row in json.loads(args.provenance.read_text(encoding="utf-8")):
            if row.get("kind") == "geotiff":
                provenance[Path(row["path"]).resolve()] = row
    records = []
    for raster in rasters:
        match = re.search(r"NAC_PHO_(E\d+[NS]\d+)_", raster.name)
        if not match:
            raise ValueError(f"cannot derive archive region from {raster.name}")
        region = match.group(1)
        label = raster.with_suffix(".xml")
        label_path = label if label.exists() else None
        row = provenance.get(raster.resolve())
        source_url = (
            row["url"]
            if row
            else (
                "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/"
                f"LROLRC_2001/EXTRAS/BROWSE/NAC_PHO/{region}/{raster.name}"
            )
        )
        records.append(
            image_record_from_lroc(raster, label_path, region_id=region, source_url=source_url)
        )
    write_manifest(records, args.output_root / "images.parquet")
    if args.preserve_splits:
        split_map = json.loads(args.preserve_splits.read_text(encoding="utf-8"))
        splits = geographic_split(records, seed=args.seed, fixed_regions=split_map)
    else:
        splits = geographic_split(records, seed=args.seed)
    write_split_manifest(splits, args.output_root / "split_regions.json")
    validate_no_leakage(splits)
    for name, split_records in splits.items():
        write_manifest(split_records, args.output_root / f"{name}.parquet")
        positives = generate_pairs(split_records)
        negatives = generate_negative_pairs(split_records)
        if positives:
            write_pairs(positives, args.output_root / "pairs" / f"{name}.parquet")
        if negatives:
            write_pairs(negatives, args.output_root / "pairs" / f"{name}_negative.parquet")
            hard = [
                replace(pair, negative_type="hard_similar_gsd")
                for pair in negatives
                if pair.relative_gsd_ratio <= 1.25
            ]
            if hard:
                write_pairs(hard, args.output_root / "pairs" / f"{name}_hard_negative.parquet")
        print(
            f"{name}: {len(split_records)} images, {len(positives)} positive, "
            f"{len(negatives)} negative"
        )
    print(f"manifest: {args.output_root / 'images.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
