# Development Handoff

## 2026-09-11 — Mac M4 offline demo release

- Build output: `release/chandrappan_mac_m4_demo/`; see `TRANSFER_MANIFEST.md` inside it.
- Live matcher: frozen Official RoMa v2 only. D1 is never loaded and remains `EXPERIMENTAL · NOT PROMOTED`.
- Public release candidate: the README now distinguishes the stored 4/4 base-640 demo result from the incomplete 2/4 strict production acceptance run. The repository keeps only compact previews and reports; raw NASA imagery, model weights, the Mac bundle, and the large image archive remain local/ignored.
- Release is relocatable and local-only: one 1.1 GiB official checkpoint, local DINOv3 Torch Hub source, 20 processed RP-001 rasters, cached descriptors/thumbnails, no raw PDS3/GT archive/training state.
- Ubuntu CUDA release checks passed: empty-host-cache offline load, held-out rank-1 verified live case, geometry-rejected hard negative, and FastAPI endpoints.
- MPS is not validated on this Ubuntu host. On the Mac run `setup_mac_m4.sh`, `doctor_mac_demo.py`, `benchmark_demo_resolution.py`, then `preflight_demo.sh`. Do not claim M4 performance beforehand.

## Branch

`dev`

## Current commit

Pending objective-correction commit

## Goal

Build the verifiable Chandrappan foundation and a scientifically honest regional production baseline. A one-off, user-authorized RP-001 D1 hackathon experiment was completed separately and did not pass either gate.

## Current state

Phase 0/GEO-001 and the available real-data geometric baseline are complete. The new training diagnostic is source-faithful at 640x640: it refuses anisotropic resizing, uses physical pixel GT sampled at every refiner stage, applies the RoMa robust loss at strides 4/2/1, and updates only refiners (including refiner-only EMA). The full suite has 65 tests.

RP-001 was built around the Apollo 15 S-IVB Impact site (`E009S3481`): 20 official full PDS3 products, exact source/hash provenance, a 12/4/4 frozen acquisition split, 100% map overlap, PDS3 backplane diversity, 162 map-derived GT artifacts, and a TRAIN-only reference bank. The official RoMa v2 precise bidirectional run is real but fails the regional gate: held-out positive VRR is `.5` (2/4), while 3 available negatives have FAR `0.0`. All NCC proposals were rejected as reverse-inconsistent or residual-degrading. See `results/RP-001_FINAL_REPORT.md`; it is explicitly **not accepted**.

The RP-001 D1 experiment checked every 12-choose-2 TRAIN combination against real masks. Nineteen pairs had exact >=95% common-valid 640px support, providing 97 crops. Its only trainable parameters were `refiners.1`; all other RoMa parameters were frozen. RAW preprocessing was selected only from TRAIN/VALIDATION. LR `3e-6`, step `10` was selected on validation, but paired metrics are neutral/slightly worse: TRAIN PCK@1 `.612917767→.612626400`; validation PCK@1 `.745905315→.741747784`; validation VRR `3/4→3/4`. TEST pairing is `.825662145→.825721564` PCK@1 and `4/4→4/4` VRR. This is not a successful regional specialist: `PRODUCTION_GATE: FAIL`, `HACKATHON_DEMO_GATE: FAIL`. Generated run/demo artifacts are ignored; never promote the static visuals beyond their stated experimental comparison.

## Known blockers

The immutable T0 smoke baseline remains untouched. The historic loss-collapse run is now classified as invalid training evidence because it optimized T0; its runner fails loudly. The non-T0 controlled robust run lowers validation median EPE (4.0009→3.8998 px), raises PCK@3/5, and preserves VRR/FAR (.5/.0), but decreases PCK@1 (.106502→.104818). `FULL TRAINING GATE: FAIL`. Details are in `results/training_objective_fix.{json,md}`. Use `/usr/bin/python3.10`, not the incomplete `/usr/local/bin/python3` build. The ignored vendored checkout has a local checkpoint-path constructor edit; preserve it.

## Next step

Do not rerun or promote the D1 checkpoint. Diagnose the true scientific-mask coverage and the two rejected official precise-protocol acquisitions without changing scientific thresholds; add independent semantic-similarity and shadow-negative labels, generate per-query visuals, and rerun the whole gate. Preserve official RoMa v2 as production. LunarMatch-NASA products/manifest, official splits/metadata, and nodata-aware/covisible GT remain separate requirements.

## Latest curation

Run `scripts/curate_scientific_lroc.py` to reproduce `runs/lroc_curated_v1` and the provenance files in `results/`. It uses only the TRAIN split, evaluates 640px overlap-centred TIFF windows for mask/cycle-valid geographic GT, and reports corrupt-raster reads as explicit rejections. The frozen manifest contains three selected pairs; it is too small and geographically narrow for a smoke-training launch.
