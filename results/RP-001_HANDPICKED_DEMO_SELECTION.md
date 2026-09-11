# RP-001 handpicked demo selection

Status: **FAIL under the requested three-VERIFIED-positive requirement.**

The frozen official RoMa v2 baseline contains four held-out TEST acquisitions, but only two pass the unchanged registration acceptance protocol. No third VERIFIED TEST backup exists. The backup slot is therefore preserved as a rejected audit candidate and is not represented as VERIFIED.

## PRIMARY

- filename: `NAC_PHO_E009S3481_M1345996066R_science.tif`
- product ID: `NAC_PHO_E009S3481_M1345996066R`
- why selected: strongest available official TEST case; 4,000/4,000 inliers, 1.0 inlier ratio, grid coverage 0.125, hull coverage 0.0670, and VERIFIED.
- expected selected reference: `NAC_PHO_E009S3481_M1118958225R`
- reference rank: 1
- PCK@1: 1.0
- median EPE: 0.4640505733 px
- VRR/verdict: VERIFIED
- runtime: not recorded in the authoritative baseline

## DIFFICULT LIGHTING

- filename: `NAC_PHO_E009S3481_M1177841115R_science.tif`
- product ID: `NAC_PHO_E009S3481_M1177841115R`
- why selected: second available VERIFIED TEST case with the largest genuine appearance change among the VERIFIED cases.
- expected selected reference: `NAC_PHO_E009S3481_M106949300R`
- reference rank: 3
- PCK@1: 1.0
- median EPE: 0.1567209734 px
- VRR/verdict: VERIFIED
- illumination differences: Δ incidence = 10.0668°, Δ emission = 8.3313°, Δ phase = 22.0709° (absolute query/reference differences)
- runtime: not recorded in the authoritative baseline

## BACKUP

- filename: `NAC_PHO_E009S3481_M177711422R_science.tif`
- product ID: `NAC_PHO_E009S3481_M177711422R`
- why retained: best auditable remaining TEST candidate, but it is rejected by unchanged geometry acceptance.
- attempted selected reference: `NAC_PHO_E009S3481_M160030722R`
- reference rank: 1
- PCK@1: 0.492
- median EPE: 1.0101983282 px
- verdict: REJECTED
- rejection reason: `low_grid_coverage;low_hull_coverage`

## HARD NEGATIVE

- filename: `NAC_PHO_E018N3346_M1142603254L.TIF`
- source region: E018N3346, outside RP-001 and geographically distinct from Apollo 15 S-IVB site E009S3481
- why challenging: real cratered LROC terrain with similar image statistics and 1.1 m/px GSD
- attempted RP-001 TRAIN reference: `NAC_PHO_E009S3481_M160030722R`
- rejection reason: official RoMa v2 correspondence geometry is rejected; no candidate passes the unchanged verification gate.

## Leakage and reproducibility

The two positive inputs are whole acquisitions from the frozen RP-001 TEST split, not TRAIN or VALIDATION products. Their SHA256 values, source URLs, metadata, and selected references are in [`demo/RP-001/handpicked/demo_image_manifest.json`](../demo/RP-001/handpicked/demo_image_manifest.json). The query-only copies and relative Mac bundle manifest are in `demo/RP-001/UPLOAD_THESE/` and `release/chandrappan_mac_m4_demo/demo_inputs/`.

`TRAINING LEAKAGE: NONE` for both positive inputs.

`DEMO IMAGE PACK: FAIL`
