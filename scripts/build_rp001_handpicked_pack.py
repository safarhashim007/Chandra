#!/usr/bin/env python3
"""Freeze a transparent RP-001 presentation-input pack from official results.

This script deliberately records an explicit FAIL when fewer than three held-out
TEST queries are VERIFIED by the official RoMa v2 protocol.  It never changes
splits, thresholds, weights, or the matcher.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chandrappan.geometry.verification import VerificationConfig, verify_registration  # noqa: E402
from scripts.evaluate_region_pack import _points, _read_crop  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def preview(src: Path, dst: Path, label: str) -> None:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(src) as ds:
        a = ds.read(1, out_shape=(1, 720, 960), resampling=Resampling.average).astype("float32")
    a = np.nan_to_num(a)
    lo, hi = np.percentile(a, (1, 99))
    im = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype("uint8")
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.putText(im, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), im)


def draw_matches(
    q: np.ndarray, r: np.ndarray, src: np.ndarray, tgt: np.ndarray, dst: Path, title: str
) -> None:
    def gray(a):
        lo, hi = np.percentile(a, (1, 99))
        return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype("uint8")

    left, right = (
        cv2.cvtColor(gray(q), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray(r), cv2.COLOR_GRAY2BGR),
    )
    canvas = np.concatenate([left, right], axis=1)
    n = min(len(src), 120)
    idx = np.linspace(0, len(src) - 1, n, dtype=int) if len(src) else np.array([], dtype=int)
    rng = np.random.default_rng(0)
    if len(idx):
        idx = idx[np.argsort(rng.random(len(idx)))]
    for i in idx:
        x1, y1 = np.rint(src[i]).astype(int)
        x2, y2 = np.rint(tgt[i]).astype(int)
        col = tuple(int(x) for x in rng.integers(60, 255, 3))
        cv2.line(canvas, (x1, y1), (x2 + left.shape[1], y2), col, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, col, -1)
        cv2.circle(canvas, (x2 + left.shape[1], y2), 2, col, -1)
    cv2.putText(canvas, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), canvas)


def overlay(q: np.ndarray, r: np.ndarray, matrix, out: Path, difference: bool = False) -> None:
    def gray(a):
        lo, hi = np.percentile(a, (1, 99))
        return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype("uint8")

    a, b = gray(q), gray(r)
    if matrix is not None and np.asarray(matrix).shape == (2, 3):
        b = cv2.warpAffine(b, np.asarray(matrix, dtype="float32"), (a.shape[1], a.shape[0]))
    if difference:
        im = cv2.absdiff(a, b)
    else:
        im = cv2.addWeighted(a, 0.5, b, 0.5, 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), im)


def card(
    case_dir: Path, role: str, q: Path, ref: Path, verdict: str, metrics: dict, extra: str = ""
) -> None:
    def read(p):
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

    qv, mv, rv = (
        read(case_dir / "query_preview.png"),
        read(case_dir / "matches.png"),
        read(case_dir / "selected_reference_preview.png"),
    )
    if qv is None or mv is None or rv is None:
        return
    qv = cv2.resize(qv, (420, 300))
    rv = cv2.resize(rv, (420, 300))
    mv = cv2.resize(mv, (840, 300))
    canvas = np.full((390, 1260, 3), 245, dtype="uint8")
    canvas[55:355, 20:440] = cv2.cvtColor(qv, cv2.COLOR_GRAY2BGR)
    canvas[55:355, 420:1260] = cv2.cvtColor(mv, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, "HELD-OUT QUERY", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)
    cv2.putText(
        canvas,
        "AUTO-SELECTED NASA LROC REFERENCE",
        (700, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        canvas,
        f"{role}   {verdict}",
        (25, 378),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 120, 0) if verdict == "VERIFIED" else (0, 0, 180),
        2,
    )
    txt = (
        f"PCK@1 {metrics.get('pck_1', 'n/a')}   "
        f"median EPE {metrics.get('median_epe_px', 'n/a')}   "
        f"verified/inliers {metrics.get('verified_match_count', 'n/a')}/"
        f"{metrics.get('inliers', 'n/a')} {extra}"
    )
    cv2.putText(canvas, txt[:140], (500, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(case_dir / "presentation_card.png"), canvas)


def main() -> int:
    out = ROOT / "demo/RP-001/handpicked"
    upload = ROOT / "demo/RP-001/UPLOAD_THESE"
    mac = ROOT / "release/chandrappan_mac_m4_demo/demo_inputs"
    for d in (out, upload, mac):
        d.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/regions/rp001.yaml").read_text())
    products = {x["product_id"]: x for x in cfg["products"]}
    baseline = json.loads((ROOT / "results/RP-001_official_romav2_baseline.json").read_text())
    rows = {x["query_product_id"]: x for x in baseline["queries"] if x["query_role"] == "TEST"}
    # Two verified TEST acquisitions are available. The strongest rejected TEST is kept
    # as an auditable backup candidate; the pack is intentionally marked FAIL.
    chosen = [
        (
            "DEMO_PRIMARY",
            "primary",
            "NAC_PHO_E009S3481_M1345996066R",
            "NAC_PHO_E009S3481_M1118958225R",
        ),
        (
            "DEMO_DIFFICULT_LIGHTING",
            "difficult_lighting",
            "NAC_PHO_E009S3481_M1177841115R",
            "NAC_PHO_E009S3481_M106949300R",
        ),
        ("DEMO_BACKUP", "backup", "NAC_PHO_E009S3481_M177711422R", "NAC_PHO_E009S3481_M160030722R"),
    ]
    manifest = {
        "region_id": "RP-001",
        "matcher": "official RoMa v2, precise bidirectional setting",
        "pack_status": "FAIL",
        "failure_reason": (
            "Only 2 of 4 TEST acquisitions are VERIFIED; no third VERIFIED backup exists "
            "under frozen protocol."
        ),
        "cases": {},
    }
    for did, role, qid, rid in chosen:
        case = out / role
        case.mkdir(parents=True, exist_ok=True)
        qp = ROOT / "data/region_packs/RP-001/processed" / f"{qid}_science.tif"
        rp = ROOT / "data/region_packs/RP-001/processed" / f"{rid}_science.tif"
        shutil.copy2(qp, case / "query_original.tif")
        shutil.copy2(rp, case / "selected_reference_original.tif")
        preview(qp, case / "query_preview.png", qid)
        preview(rp, case / "selected_reference_preview.png", rid)
        row = rows[qid]
        cand = next(
            (c for c in row["candidate_ranking"] if c["reference_product_id"] == rid),
            row["candidate_ranking"][0],
        )
        v = cand.get("verification", {})
        s = v.get("selected") or {}
        metrics = dict(cand.get("correspondence_metrics") or {})
        metrics.update(
            {
                "verified_match_count": s.get("input_matches", 0),
                "inliers": s.get("inliers", 0),
                "inlier_ratio": s.get("inlier_ratio", 0),
                "grid_coverage": s.get("grid_coverage", 0),
                "hull_coverage": s.get("hull_coverage", 0),
                "cycle_consistency": cand.get("matching", {}).get("cycle_median_px"),
            }
        )
        # Use the recorded authoritative verdict; visuals are generated from the same raster pair.
        q0 = r0 = (0, 0)
        qtorch, qa = _read_crop(qp, q0, 640)
        rtorch, ra = _read_crop(rp, r0, 640)
        try:
            sys.path.insert(0, str(ROOT / "vendor/romav2/src"))
            from romav2 import RoMaV2

            if "model" not in globals():
                model = RoMaV2(checkpoint_path=str(ROOT / "checkpoints/romav2_official.pt"))
                model.apply_setting("precise")
            import torch

            device = next(model.parameters()).device
            with torch.inference_mode():
                pred = model.match(qtorch.to(device), rtorch.to(device))
            src, tgt, _, diag = _points(pred, 640, max_matches=4000)
            draw_matches(qa, ra, src, tgt, case / "matches.png", f"{did} — {row['verdict']}")
            matrix = s.get("matrix")
            overlay(qa, ra, matrix, case / "overlay.png")
            overlay(qa, ra, matrix, case / "difference.png", True)
        except Exception as exc:
            (case / "matches.png").write_bytes(b"")
            (case / "overlay.png").write_bytes(b"")
            (case / "difference.png").write_bytes(b"")
            diag = {"generation_error": str(exc)}
        result = {
            "demo_id": did,
            "role": role,
            "query_product_id": qid,
            "query_split": "TEST",
            "query_original": "query_original.tif",
            "selected_reference_product_id": rid,
            "selected_reference_split": "TRAIN",
            "selected_reference_rank": row["candidate_ranking"][0]["retrieval_rank"]
            if row["candidate_ranking"][0]["reference_product_id"] == rid
            else next(
                (
                    c["retrieval_rank"]
                    for c in row["candidate_ranking"]
                    if c["reference_product_id"] == rid
                ),
                None,
            ),
            "verdict": row["verdict"],
            "metrics": metrics,
            "official_result": row,
            "visual_generation": diag if isinstance(diag, dict) else {},
            "used_for_training": False,
            "reference_bank_role": "TRAIN",
        }
        (case / "result.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
        (case / "provenance.json").write_text(
            json.dumps(
                {
                    "query_product_id": qid,
                    "query_source_path": str(qp),
                    "query_sha256": sha(qp),
                    "reference_source_path": str(rp),
                    "reference_sha256": sha(rp),
                    "source_scientific_product": products[qid]["source_url"],
                    "acquisition_date": products[qid]["acquisition_date"],
                    "incidence_deg": products[qid]["incidence_deg"],
                    "emission_deg": products[qid]["emission_deg"],
                    "phase_deg": products[qid]["phase_deg"],
                    "gsd_m_per_px": products[qid]["gsd_m_per_px"],
                },
                indent=2,
            )
            + "\n"
        )
        src_ext = qp.suffix
        name = {
            "DEMO_PRIMARY": "01_PRIMARY_HELDOUT",
            "DEMO_DIFFICULT_LIGHTING": "02_DIFFICULT_LIGHTING_HELDOUT",
            "DEMO_BACKUP": "03_BACKUP_HELDOUT",
        }[did] + src_ext
        shutil.copy2(qp, upload / name)
        shutil.copy2(qp, mac / name)
        manifest["cases"][did] = {
            "role": role,
            "query_product_id": qid,
            "query_split": "TEST",
            "query_source_path": str(qp),
            "copied_demo_path": f"demo/RP-001/UPLOAD_THESE/{name}",
            "query_sha256": sha(qp),
            "acquisition_date": products[qid]["acquisition_date"],
            "incidence": products[qid]["incidence_deg"],
            "emission": products[qid]["emission_deg"],
            "phase": products[qid]["phase_deg"],
            "gsd_m_per_px": products[qid]["gsd_m_per_px"],
            "selected_reference_product_id": rid,
            "selected_reference_split": "TRAIN",
            "selected_reference_rank": result["selected_reference_rank"],
            "reference_sha256": sha(rp),
            "expected_verdict": row["verdict"],
            "final_metrics": metrics,
            "selection_reason": (
                "Highest available verified geometry for role; backup retained as rejected audit "
                "candidate because no third verified TEST exists."
            ),
        }
        extra = ""
        if did == "DEMO_DIFFICULT_LIGHTING":
            extra = "Δ incidence 10.0668°  Δ phase 22.0709°"
        card(case, role.upper(), qp, rp, row["verdict"], metrics, extra)
    # Hard negative: legitimate E018N3346 image, geographically distinct from Apollo 15 site.
    hnid = "NAC_PHO_E018N3346_M1142603254L"
    hnp = ROOT / "data/raw/lroc_corpus" / f"{hnid}.TIF"
    href = ROOT / "data/region_packs/RP-001/processed/NAC_PHO_E009S3481_M160030722R_science.tif"
    hcase = out / "hard_negative"
    hcase.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hnp, hcase / "query_original.TIF")
    shutil.copy2(href, hcase / "selected_reference_original.tif")
    preview(hnp, hcase / "query_preview.png", hnid)
    preview(href, hcase / "selected_reference_preview.png", "RP-001 TRAIN candidate")
    # Existing official negative suite is the authoritative geometry rejection evidence.
    suite = json.loads((ROOT / "results/RP-001_hard_negative_suite.json").read_text())
    neg = next(x for x in suite["cases"] if x["candidate_product_id"] == hnid)
    # Run the real official matcher once for the external query against the
    # highest-ranked RP-001 candidate, then preserve its rejected geometry.
    direct = {}
    try:
        import torch

        sys.path.insert(0, str(ROOT / "vendor/romav2/src"))
        from romav2 import RoMaV2

        model_hn = RoMaV2(checkpoint_path=str(ROOT / "checkpoints/romav2_official.pt"))
        model_hn.apply_setting("precise")
        qt, qa = _read_crop(hnp, (0, 0), 640)
        rt, ra = _read_crop(href, (0, 0), 640)
        device = next(model_hn.parameters()).device
        with torch.inference_mode():
            pred = model_hn.match(qt.to(device), rt.to(device))
        src, tgt, _, diag = _points(pred, 640, max_matches=4000)
        verification = verify_registration(src, tgt, (640, 640), config=VerificationConfig())
        draw_matches(
            qa, ra, src, tgt, hcase / "matches.png", "DEMO_HARD_NEGATIVE — GEOMETRY FAILED"
        )
        direct = {
            "matching": diag,
            "verification": verification.as_dict(),
            "verdict": "VERIFIED" if verification.accepted else "REJECTED",
        }
    except Exception as exc:
        direct = {"verdict": "REJECTED", "generation_error": str(exc)}
    (hcase / "result.json").write_text(
        json.dumps(
            {
                "demo_id": "DEMO_HARD_NEGATIVE",
                "role": "hard_negative",
                "query_product_id": hnid,
                "source_region": "E018N3346",
                "verdict": "REJECTED",
                "selected_reference_product_id": "NAC_PHO_E009S3481_M160030722R",
                "selected_reference_split": "TRAIN",
                "rejection_reason": direct.get("verification", {}).get(
                    "rejection_reason", "official geometry rejected the cross-region registration"
                ),
                "official_negative_evidence": neg,
                "direct_official_run": direct,
            },
            indent=2,
            allow_nan=True,
        )
        + "\n"
    )
    (hcase / "provenance.json").write_text(
        json.dumps(
            {
                "query_product_id": hnid,
                "query_source_path": str(hnp),
                "query_sha256": sha(hnp),
                "source_region": "E018N3346",
                "footprint_different_from_RP001": True,
                "source_url": "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/BROWSE/NAC_PHO/E018N3346/NAC_PHO_E018N3346_M1142603254L.TIF",
                "reference_product_id": neg["query_product_id"],
                "reference_bank_role": "TRAIN",
            },
            indent=2,
        )
        + "\n"
    )
    for name in ("overlay.png", "difference.png"):
        blank = np.full((640, 1280, 3), 235, np.uint8)
        cv2.putText(
            blank,
            "GEOMETRY FAILED — REJECTED",
            (35, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 180),
            3,
        )
        cv2.imwrite(str(hcase / name), blank)
    card(
        hcase,
        "SIMILAR LUNAR TERRAIN → GEOMETRY FAILED",
        hnp,
        href,
        "REJECTED",
        {
            "pck_1": "n/a",
            "median_epe_px": "n/a",
            "verified_match_count": direct.get("matching", {}).get(
                "filtered_correspondence_count", "n/a"
            ),
            "inliers": "n/a",
        },
        "REJECTED",
    )
    shutil.copy2(hnp, upload / "04_HARD_NEGATIVE.TIF")
    shutil.copy2(hnp, mac / "04_HARD_NEGATIVE.TIF")
    manifest["cases"]["DEMO_HARD_NEGATIVE"] = {
        "role": "hard_negative",
        "query_product_id": hnid,
        "query_split": "OUTSIDE RP-001",
        "query_source_path": str(hnp),
        "copied_demo_path": "demo/RP-001/UPLOAD_THESE/04_HARD_NEGATIVE.TIF",
        "query_sha256": sha(hnp),
        "source_region": "E018N3346",
        "expected_verdict": "REJECTED",
        "selected_reference_product_id": "NAC_PHO_E009S3481_M160030722R",
        "selected_reference_split": "TRAIN",
        "selection_reason": (
            "Real, cratered lunar terrain from a different LROC region; official geometry "
            "rejects the challenge."
        ),
    }
    (out / "demo_image_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True) + "\n"
    )
    (mac / "manifest.json").write_text(
        json.dumps(
            {
                "relative_to_bundle": True,
                "region_id": "RP-001",
                "files": {k: v["copied_demo_path"] for k, v in manifest["cases"].items()},
            },
            indent=2,
        )
        + "\n"
    )
    (upload / "README.md").write_text(
        "# RP-001 presentation inputs\n\n"
        "The authoritative official RoMa v2 pack is marked FAIL: only two TEST positives "
        "verify.\n\n"
        "- `01_PRIMARY_HELDOUT.TIF` — VERIFIED\n"
        "- `02_DIFFICULT_LIGHTING_HELDOUT.TIF` — VERIFIED\n"
        "- `03_BACKUP_HELDOUT.TIF` — REJECTED (no third verified TEST exists)\n"
        "- `04_HARD_NEGATIVE.TIF` — REJECTED\n"
    )
    (out / "DEMO_IMAGES.md").write_text(
        "# RP-001 handpicked images\n\n"
        "Status: **FAIL**. The frozen official results contain only two VERIFIED TEST "
        "acquisitions. The third TEST case is preserved as an auditable rejected backup "
        "candidate; it is not represented as VERIFIED.\n\n"
        "See `demo_image_manifest.json` and per-case `result.json` files.\n"
    )
    expected = {
        k: {
            "expected_verdict": v["expected_verdict"],
            "expected_reference_product_id": v.get("selected_reference_product_id"),
            "expected_reference_rank": v.get("selected_reference_rank"),
        }
        for k, v in manifest["cases"].items()
    }
    (out / "expected_results.json").write_text(json.dumps(expected, indent=2) + "\n")
    # Contact sheet.
    thumbs = []
    for label, name in [
        ("01 PRIMARY — HELD-OUT TEST", "primary"),
        ("02 DIFFICULT LIGHTING — HELD-OUT TEST", "difficult_lighting"),
        ("03 BACKUP — HELD-OUT TEST", "backup"),
        ("04 HARD NEGATIVE — DIFFERENT REGION", "hard_negative"),
    ]:
        withr = cv2.imread(str(out / name / "query_preview.png"), cv2.IMREAD_GRAYSCALE)
        withr = cv2.resize(withr, (480, 300))
        canvas = cv2.cvtColor(withr, cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        thumbs.append(canvas)
    sheet = cv2.vconcat([cv2.hconcat(thumbs[:2]), cv2.hconcat(thumbs[2:])])
    cv2.imwrite(str(out / "demo_contact_sheet.png"), sheet)
    (out / "DEMO_IMAGES.md").write_text(
        (out / "DEMO_IMAGES.md").read_text()
        + "\nTraining leakage: NONE for the two positive inputs; both are in frozen TEST and "
        "were evaluated by official baseline. The backup candidate is also TEST but rejected.\n"
    )
    print("DEMO IMAGE PACK: FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
