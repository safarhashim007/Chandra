#!/usr/bin/env python3
"""Audit the existing LROC corpus before choosing a Region Pack site.

This report is intentionally conservative: a missing angle, original-acquisition
time, or overlap measurement is recorded as missing.  It is never replaced with
a proxy from a PDS browse-product creation timestamp.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.data.manifest import read_manifest
from chandrappan.data.pairs import generate_pairs


def _range(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": min(values), "max": max(values), "span": max(values) - min(values)}


def _curation_by_region(path: Path) -> dict[str, dict[str, int | float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("region_pair_yield", {})


def _candidate_score(row: dict[str, object]) -> float:
    """Rank using only evidence available at audit time; angles are not guessed."""
    discovery = row["official_discovery"]
    local = row["local_corpus"]
    curation = row["curation"]
    assert isinstance(discovery, dict)
    assert isinstance(local, dict)
    assert isinstance(curation, dict)
    observation = min(float(discovery["product_count"]), 100.0) / 100.0
    overlap = float(local["median_overlap_ratio"] or 0.0)
    valid_yield = float(curation.get("pair_yield", 0.0))
    map_quality = 1.0 if local["map_projected_valid_count"] == local["product_count"] else 0.0
    gsd = local["gsd_m_per_px"]
    gsd_span = float(gsd["span"]) if isinstance(gsd, dict) else 0.0
    # A modest bonus rewards actual GSD variety while repeat count and real
    # overlap remain decisive.  Unknown illumination never receives points.
    return round(
        55 * observation + 25 * overlap + 10 * valid_yield + 7 * map_quality + min(gsd_span, 3),
        6,
    )


def build_audit(
    manifest_path: Path, train_manifest_path: Path, discovery_path: Path, curation_path: Path
) -> dict[str, object]:
    records = read_manifest(manifest_path)
    train_records = read_manifest(train_manifest_path)
    discovery_rows = json.loads(discovery_path.read_text(encoding="utf-8"))
    discovery = {row["site_id"]: row for row in discovery_rows}
    curation = _curation_by_region(curation_path)
    grouped = defaultdict(list)
    for record in records:
        grouped[record.region_id].append(record)

    rows: list[dict[str, object]] = []
    for site_id, source in discovery.items():
        site_records = grouped.get(site_id, [])
        pairs = generate_pairs(site_records, min_overlap_ratio=0.01)
        overlaps = [pair.overlap_ratio for pair in pairs]
        incidence = [row.incidence_angle for row in site_records if row.incidence_angle is not None]
        emission = [row.emission_angle for row in site_records if row.emission_angle is not None]
        phase = [row.phase_angle for row in site_records if row.phase_angle is not None]
        gsds = [row.gsd_m_per_px for row in site_records]
        paths = [Path(row.image_path) for row in site_records]
        row: dict[str, object] = {
            "site_id": site_id,
            "official_discovery": {
                "product_count": source["product_count"],
                "candidate_positive_pairs": source["candidate_positive_pairs"],
                "geotiff_kbytes_total": source["geotiff_kbytes_total"],
                "product_img_kbytes_total": source["product_img_kbytes_total"],
                "all_products_have_masks": source["has_masks"],
                "acquisition_priority": source["acquisition_priority"],
            },
            "local_corpus": {
                "product_count": len(site_records),
                "products": [
                    record.product_id for record in sorted(site_records, key=lambda x: x.product_id)
                ],
                "map_projected_valid_count": sum(path.exists() for path in paths),
                "positive_pairs_with_at_least_1pct_overlap": len(pairs),
                "median_overlap_ratio": median(overlaps) if overlaps else None,
                "minimum_overlap_ratio": min(overlaps) if overlaps else None,
                "maximum_overlap_ratio": max(overlaps) if overlaps else None,
                "gsd_m_per_px": _range(gsds),
                "incidence_deg": _range([float(value) for value in incidence]),
                "emission_deg": _range([float(value) for value in emission]),
                "phase_deg": _range([float(value) for value in phase]),
                "angle_metadata_coverage": {
                    "incidence": len(incidence) / len(site_records) if site_records else 0.0,
                    "emission": len(emission) / len(site_records) if site_records else 0.0,
                    "phase": len(phase) / len(site_records) if site_records else 0.0,
                },
                "reported_acquisition_times": sorted(
                    {record.acquisition_time for record in site_records if record.acquisition_time}
                ),
            },
            "curation": curation.get(
                site_id,
                {
                    "candidate_positive_pairs": 0,
                    "dense_gt_valid_pairs": 0,
                    "selected_pairs": 0,
                    "pair_yield": 0.0,
                },
            ),
        }
        row["candidate_score"] = _candidate_score(row)
        rows.append(row)

    ranked = sorted(rows, key=lambda row: (-float(row["candidate_score"]), str(row["site_id"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return {
        "audit_version": "RP-001-candidate-audit-v1",
        "inputs": {
            "manifest": str(manifest_path),
            "train_manifest": str(train_manifest_path),
            "site_discovery": str(discovery_path),
            "curation": str(curation_path),
        },
        "scientific_limitations": [
            "The local full-browse GeoTIFFs are one-band browse representations.",
            "Illumination/backplane statistics require the official four-band PDS IMG files.",
            "A browse-product creation time is not substituted for source acquisition time.",
            "Scores rank acquisition candidates only; they are not registration performance "
            "claims.",
        ],
        "current_train_regions": sorted(
            {record.region_id for record in train_records if record.product_id}
        ),
        "candidates": ranked,
    }


def _markdown(audit: dict[str, object]) -> str:
    candidates = audit["candidates"]
    assert isinstance(candidates, list)
    lines = [
        "# Region Pack candidate audit",
        "",
        "This is a pre-selection audit. It ranks repeat-observation sites from the official ODE "
        "discovery response and measures only metadata present in the existing local corpus.",
        "",
        "## Scientific limitations",
        "",
        *[f"- {item}" for item in audit["scientific_limitations"]],
        "",
        "## Ranked candidates",
        "",
        "| Rank | Site | Official repeat products | Local products | Median overlap | "
        "Dense-GT valid | GSD span (m/px) | Angle coverage | Score |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidates:
        assert isinstance(row, dict)
        official = row["official_discovery"]
        local = row["local_corpus"]
        curation = row["curation"]
        assert isinstance(official, dict) and isinstance(local, dict) and isinstance(curation, dict)
        gsd = local["gsd_m_per_px"]
        gsd_span = f"{gsd['span']:.3f}" if isinstance(gsd, dict) else "unknown"
        coverage = local["angle_metadata_coverage"]
        assert isinstance(coverage, dict)
        angle_coverage = "/".join(
            f"{coverage[key]:.0%}" for key in ("incidence", "emission", "phase")
        )
        overlap = local["median_overlap_ratio"]
        lines.append(
            (
                "| {rank} | {site} | {official_count} | {local_count} | {overlap} | "
                "{valid} | {gsd} | {angles} | {score:.3f} |"
            ).format(
                rank=row["rank"],
                site=row["site_id"],
                official_count=official["product_count"],
                local_count=local["product_count"],
                overlap=f"{overlap:.3f}" if overlap is not None else "n/a",
                valid=curation.get("dense_gt_valid_pairs", 0),
                gsd=gsd_span,
                angles=angle_coverage,
                score=float(row["candidate_score"]),
            )
        )
    lines.extend(["", "## Current curation contribution", ""])
    for row in candidates:
        assert isinstance(row, dict)
        curation = row["curation"]
        assert isinstance(curation, dict)
        if curation.get("selected_pairs", 0):
            lines.append(
                f"- `{row['site_id']}`: {curation['selected_pairs']} selected / "
                f"{curation['dense_gt_valid_pairs']} dense-GT-valid local TRAIN pairs."
            )
    lines.extend(
        [
            "",
            "No site is selected by this report alone. The selected site's PDS3 source headers and "
            "angle backplanes must be acquired and validated before RP-001 can satisfy its "
            "acceptance gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/expanded/images.parquet"))
    parser.add_argument("--train-manifest", type=Path, default=Path("data/expanded/train.parquet"))
    parser.add_argument(
        "--site-discovery", type=Path, default=Path("results/lroc_site_discovery.json")
    )
    parser.add_argument(
        "--curation", type=Path, default=Path("results/dataset_curation_lroc_curated_v2.json")
    )
    parser.add_argument(
        "--json", type=Path, default=Path("results/region_pack_candidate_audit.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("results/region_pack_candidate_audit.md")
    )
    args = parser.parse_args()
    audit = build_audit(args.manifest, args.train_manifest, args.site_discovery, args.curation)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(audit), encoding="utf-8")
    print(
        json.dumps({"top_candidate": audit["candidates"][0]["site_id"], "output": str(args.json)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
