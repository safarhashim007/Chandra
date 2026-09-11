# Chandrappan Project State

Last updated: 2026-09-11 Asia/Kolkata

Current version: v0.1 release candidate

Current stable commit: pending public-release commit

Integration branch: `dev`

Primary production matcher: RoMa v2 (official checkpoint loaded and GPU smoke-tested)

Experimental matcher: LunarRoMa (not started)

## Current phase

Phase 2 dataset/evaluation foundation is complete for the available real products. RP-001 now has an acquired, validated 20-observation regional corpus, but its production acceptance gate is failed/incomplete: held-out positive VRR is 0.5 under the official precise protocol, NCC has no non-degrading accepted controls, and the full semantic/shadow negative suite is not available. A user-authorized hackathon D1 regional-specialist experiment was run separately: it used only the stride-1 refiner and the frozen 12 TRAIN acquisitions, but did not improve monitored TRAIN/VALIDATION PCK@1. Both `PRODUCTION_GATE` and `HACKATHON_DEMO_GATE` are therefore `FAIL`. MATCH-001 and TRAIN-001 remain blocked for production claims. The public RoMa v2 source checkout is at `vendor/romav2`, commit `95c9968145c8906b7b59383258e9f73b02853d89`, and the ignored official v2.0.1 checkpoint is available locally. Supported local interpreter is `/usr/bin/python` (Python 3.10).

## Completed

- [x] Empty workspace initialized as a `dev` Git repository.
- [x] Environment inspected: Linux x86_64, Python 3.14.2, NVIDIA RTX 4000 SFF Ada (20,475 MiB), 62 GiB RAM, 86 GiB free disk.
- [x] Official RoMa v2 source cloned for inspection.
- [x] Downloaded an ignored public LROC WAC map-projected GeoTIFF/XML fixture and independently verified its GeoTIFF affine tags and Moon center.

## In progress

- [x] Repository-management infrastructure and Phase 0 tools.
- [x] Moon-specific longitude conventions, affine pixel/world transforms, metadata contract, SQLite/RTree catalog implementation, and synthetic roundtrip tests.
- [x] Real-product tag-level affine roundtrip checked for `WAC_TIO2_E350N0450` with maximum error `2.6e-13 px`.
- [x] Rasterio/pyproj LROC adapter validated against the WAC fixture; all project tests pass with the real-product test enabled.
- [x] Two real map-projected NAC photometric products were ingested into an ignored SQLite/RTree catalog and their footprints were verified to overlap.
- [x] Official RoMa v2.0.1 checkpoint loaded; turbo and base CUDA inference smoke tests returned finite normalized warps and confidence tensors.
- [x] Separate gradient-enabled training forward implemented; a CUDA gradient smoke test reached the refiner parameters while frozen descriptor/matcher parameters stayed without gradients.
- [x] Pixel-centre GT and official RoMa coordinate semantics are covered by regression tests.
- [x] Canonical Parquet image manifest, deterministic geographic splits, leakage checks, and footprint-overlap pair generation implemented.
- [x] Project-defined one-pair T0 benchmark frozen and SHA256 checked; untouched RoMa v2 baseline recorded on CUDA.
- [x] VRR/FAR acceptance definitions and a hard validation quality gate implemented; T0 is rejected from training by path or checksum.
- [x] Training objective audit completed: beta=1.0 is 320 px/axis at 640, while beta=.01 is 3.2 px/axis; coordinate and displacement round trips pass on non-square grids.
- [x] Corrected refiner-only path uses source RoMa robust EPE loss at strides 4/2/1, stage-wise GT sampling, refiner-only EMA, frozen-parameter checks, T0-pair rejection, and a source-faithful 640x640 input contract.
- [x] Bounded CUDA LR ablation completed on a geographically TRAIN-only LROC pair and evaluated on held-out validation. The conservative run improves median EPE/PCK@3/PCK@5 but slightly decreases PCK@1, so the strict training gate remains failed.
- [x] Expanded genuine LROC NAC corpus built: 20 images across 7 geographic regions, with leakage-safe train/validation/test manifests and positive/negative pair manifests.
- [x] Untouched RoMa v2 validation baseline recorded: 2 positive pairs, 4 negatives, VRR 1.0, FAR 0.0 under the selected project configuration; confidence and outlier distributions recorded.
- [x] RP-001 candidate audit, source-header verification, and raw NASA acquisition completed for the Apollo 15 S-IVB Impact site (`E009S3481`): 20 exact acquisitions, 12/4/4 acquisition-isolated split, SHA256 provenance, 100% canonical map-footprint overlap, and measured PDS3 backplane diversity (incidence 4.497–83.554°, emission 2.074–27.820°, phase 1.906–81.933°).
- [x] RP-001 TRAIN-only deterministic reference bank, 162 map-derived GT pair artifacts, contact sheets, read-only region metadata endpoints, and untouched official RoMa v2 precise bidirectional baseline are implemented and executed. The baseline's held-out positive VRR is 0.5; three evaluated negative cases have FAR 0.0.
- [x] User-authorized RP-001 hackathon D1 experiment: all 66 TRAIN combinations were checked against scientific masks, yielding 19 GT-valid pairs and 97 exact 640px TRAIN crops; 47 pair exclusions are explicit no-nodata-aware-window rejections. RAW/robust/CLAHE/gamma preprocessing was compared on TRAIN/VALIDATION only; RAW won. D1 froze backbone/DINO, matcher, stride-4 and stride-2 refiners and ran 1e-6/3e-6/1e-5 through steps 10/25/50/100/250. The selected EMA state is 3e-6, step 10. Paired base-640 metrics are neutral/slightly worse on TRAIN/VALIDATION PCK@1, so it is `EXPERIMENTAL_REGION_SPECIALIST`, not a demo or production pass.
- [x] An ignored local RP-001 demo bundle was rendered with clearly labeled precomputed TRAIN/held-out-validation match views, overlays, difference/blink assets, rejected other-region negative, training provenance, checkpoint data card, static frontend, and an offline Mac inference model cache. It is explicitly stamped `HACKATHON_DEMO_GATE: FAIL`.
- [x] Mac M4 offline presentation release assembled at `release/chandrappan_mac_m4_demo`: frozen Official RoMa v2, pinned local DINOv3 source, 20 processed RP-001 rasters, cached descriptors/thumbnails, local FastAPI frontend/backend, transfer checksums, and setup/start/doctor/preflight scripts. Raw PDS3, GT archive, training crops, optimizer state, CUDA environment, and D1 runtime weights are excluded. Ubuntu CUDA live/API checks pass; physical MPS validation remains pending.
- [x] Public-release documentation, compact demo/social previews, repository description/topics, and portable setup guidance prepared. `make format`, `make lint`, `make test` (65 passed), and `python scripts/doctor.py` pass. Raw NASA products, checkpoints, Mac bundle, and large demo archive remain ignored.

## Blockers

- A proper LunarMatch-NASA training manifest, geographic train/validation/test split, and held-out benchmark are not available.
- The immutable project-defined T0 remains a one-pair smoke benchmark; the expanded LROC corpus is a project validation corpus, not official LunarMatch-NASA.
- Full training remains blocked: the best controlled run has validation PCK@1 `.104818` versus untouched `.106502`, despite improving median EPE `4.0009→3.8998 px`, PCK@3 `.3479→.3680`, PCK@5 `.5226→.5253`, and retaining VRR/FAR `.5/.0`.
- Validation geometry is available, but the current T0 result remains positive-only and is not used for threshold selection or FAR.
- `/usr/local/bin/python3` is Python 3.14 and lacks `_sqlite3`; project commands must use `/usr/bin/python` (Python 3.10).
- The vendored RoMa checkout has an existing local constructor/checkpoint-path modification; it is ignored and must not be overwritten or committed without explicit ownership review.
- RP-001 is not a successful flagship pack: two of four held-out positive queries are rejected, no local NCC proposal was non-degrading, and independent semantic-similarity/shadow hard-negative labels are unavailable. Do not present the partial FAR result as a complete safety evaluation.
- The sanctioned RP-001 D1 adaptation has no useful monitored gain: official versus D1 validation PCK@1 is `.745905315→.741747784`, validation VRR is `3/4→3/4`, and TRAIN PCK@1 is `.612917767→.612626400`. Do not relabel the static assets as evidence of a successful specialist.

## Last commands

`make format`; `make lint`; `make test` (65 passed); and `python scripts/doctor.py` all passed after the public-release documentation and lint cleanup. The D1 run used CUDA and completed cleanly with frozen-parameter assertions.

## Next exact task

Transfer `release/chandrappan_mac_m4_demo` to the MacBook Air, then run `./scripts/setup_mac_m4.sh`, `./scripts/doctor_mac_demo.py`, `./scripts/benchmark_demo_resolution.py`, and `./scripts/preflight_demo.sh`. Do not claim MPS timing or parity until they pass. Do not promote or rerun the failed D1 checkpoint. For production, diagnose real nodata-mask support and the two official precise-protocol rejections without changing acceptance thresholds; add validated shadow and crater-semantic labels and a broader corpus before any new specialist claim.

## Latest curation run

`6d894b6` created `runs/lroc_curated_v1` from the existing genuine 20-product NAC_PHO TIFF/XML corpus. Twelve TRAIN positive pairs passed 640px overlap-window dense-GT/mask/cycle validation; three pairs from one available TRAIN region were selected and SHA256-frozen. The archive expansion target was not met, and one raster read failure was classified as a rejection. Full fine-tuning was not launched.

## Latest RP-001 D1 experiment

`runs/rp001_demo_adaptation_v1` is intentionally ignored because it contains generated crop GT and checkpoint states. Its immutable pair manifest SHA256 is `b9a68b0c913e7059fc2bcc0ca93c89ba54fff207c1fb64268fd4c5ec5830c64e`. The selected state SHA256 is `11a443dda177d8a96611b96d8d57190aacc83f49f0307733e2c4dc14231e5610`. See ignored `paired_evaluation.json` and demo data card for the exact paired table. TEST was first evaluated after freeze for dense metrics, then explicitly re-evaluated at the user's request to obtain paired geometric VRR; it was never used to select preprocessing, LR, step, or checkpoint.
