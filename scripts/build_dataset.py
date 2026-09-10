#!/usr/bin/env python3
"""Build manifests, geographic splits, pairs, and a frozen project-defined T0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.manifest import image_record_from_lroc, write_manifest
from chandrappan.data.pairs import generate_negative_pairs, generate_pairs, write_pairs
from chandrappan.data.splits import geographic_split, validate_no_leakage
from chandrappan.evaluation.t0 import freeze_checksum, verify_checksum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rasters", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--t0", type=Path, default=Path("benchmarks/T0_v1.parquet"))
    parser.add_argument("--t0-sha256", type=Path, default=Path("benchmarks/T0_v1.sha256"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.01)
    parser.add_argument("--write-negatives", action="store_true")
    parser.add_argument("--skip-t0", action="store_true")
    args = parser.parse_args()

    records = [
        image_record_from_lroc(raster, raster.with_suffix(".xml")) for raster in args.rasters
    ]
    manifest_path = args.output_root / "images.parquet"
    write_manifest(records, manifest_path)
    splits = geographic_split(records, seed=args.seed)
    validate_no_leakage(splits)
    for name, split_records in splits.items():
        write_manifest(split_records, args.output_root / f"{name}.parquet")
        pairs = generate_pairs(split_records, min_overlap_ratio=args.min_overlap_ratio)
        pair_path = args.output_root / "pairs" / f"{name}.parquet"
        if pairs:
            write_pairs(pairs, pair_path)
        elif pair_path.exists():
            pair_path.unlink()
        if args.write_negatives:
            negative_path = args.output_root / "pairs" / f"{name}_negative.parquet"
            negative_pairs = generate_negative_pairs(split_records)
            if negative_pairs:
                write_pairs(negative_pairs, negative_path)
            elif negative_path.exists():
                negative_path.unlink()
        print(f"{name}: {len(split_records)} images, {len(pairs)} pairs")

    if args.skip_t0:
        print("T0: skipped by request; existing T0 was not modified")
        return 0
    test_pairs = generate_pairs(splits["test"], min_overlap_ratio=args.min_overlap_ratio)
    if not test_pairs:
        raise RuntimeError("cannot create T0: test split has no overlapping pair")
    if args.t0.exists():
        if not args.t0_sha256.exists():
            raise RuntimeError(f"existing T0 has no checksum: {args.t0}")
        verify_checksum(args.t0, args.t0_sha256)
    else:
        write_pairs(test_pairs, args.t0)
        freeze_checksum(args.t0, args.t0_sha256)
    print(f"manifest: {manifest_path}")
    print(f"T0: {args.t0} ({len(test_pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
