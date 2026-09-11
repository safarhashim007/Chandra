# Chandrappan Tasks

## REPO-001

Status: DONE
Priority: HIGH  
Goal: Establish reproducible repository workflow, health checks, documentation, and CPU-safe tests.

Acceptance: state/handoff files, doctor/system checks, format/lint/test commands, and initial commit.

## GEO-001

Status: DONE
Priority: HIGH
Depends on: REPO-001
Goal: Validate the implemented Moon-specific CRS, pixel/world transforms, footprint metadata, and catalog against real LROC map-projected products.

Evidence: the ignored WAC fixture is recorded in `data/manifests/lroc_fixture.json` and passes the rasterio-backed adapter/affine regression. The ignored NAC photometric pair is recorded in `data/manifests/lroc_nac_pair.json`, loads through the same adapter, has overlapping world footprints, and is queryable in the SQLite/RTree catalog.

## MATCH-001

Status: DONE
Priority: HIGH
Depends on: GEO-001
Goal: Establish official RoMa v2 baseline adapter and inspect actual tensor conventions.

Evidence: official v2.0.1 checkpoint loads on CUDA; turbo and base inference smoke tests return finite outputs. A held-out T0 benchmark is still unavailable.

## TRAIN-001

Status: BLOCKED
Priority: HIGH
Depends on: MATCH-001, LunarMatch-NASA data, T0 benchmark, one-batch overfit
Goal: Add a validated, source-compatible LunarRoMa training path without modifying official inference.

Evidence: a source-faithful 640x640 refiner-only path now supervises all detached strides (4/2/1) with RoMa's robust EPE loss, limits EMA to refiners, rejects T0 pair IDs, and validates only on the held-out LROC split. The strict gate remains failed because PCK@1 regressed slightly despite EPE/PCK@3/PCK@5 improvement.

## DATA-001

Status: DONE
Priority: HIGH
Depends on: GEO-001
Goal: Build validated canonical manifests, geographic splits, leakage checks, and geographic pair generation.

Evidence: Parquet manifest/split/pair smoke artifacts are generated from one WAC and two NAC products; 23 tests pass.

## BENCH-001

Status: IN_PROGRESS
Priority: HIGH
Depends on: DATA-001, MATCH-001
Goal: Freeze a project-defined T0 and record the untouched official RoMa v2 baseline.

Evidence: `benchmarks/T0_v1.parquet` is checksummed and evaluated on CUDA. Baseline metrics and error/confidence distributions are stored under `results/T0/`. This remains a one-pair project-defined benchmark, not official LunarMatch-NASA data.

## GEOM-001

Status: DONE
Priority: HIGH
Depends on: BENCH-001
Goal: Verify geometric registrations, measure coverage, create real negative pairs, and establish a real validation baseline.

Evidence: OpenCV-backed similarity/affine/homography verification, grid/hull coverage, 20-image/7-region LROC corpus, geographically isolated validation with two positives and four negatives, validation-only threshold selection, confidence/outlier analysis, optional diagnostic PNGs, and tests are implemented. Untouched RoMa validation baseline is VRR 1.0 and FAR 0.0 under the selected project configuration.

## GATE-001

Status: BLOCKED
Priority: HIGH
Depends on: BENCH-001, TRAIN-001
Goal: Prevent full training unless short validation training genuinely improves correspondence quality.

Evidence: hard gate and preflight are implemented and tested. The available NAC diagnostic fails the gate: beta=1.0 changes median EPE 1.5818→11.6061 px and PCK@1 0.4697→0.0002; beta=0.01 avoids collapse but still changes median EPE 1.5818→1.6095 px and PCK@1 0.4697→0.4611. Full fine-tuning remains blocked.

The historical diagnostic used T0 and is now retired as invalid training evidence. The corrected non-T0 robust run improves validation median EPE 4.0009→3.8998 px, PCK@3 .3479→.3680, and PCK@5 .5226→.5253, but PCK@1 .10650→.10482; the gate therefore remains blocked.

## RP-001

Status: IN_PROGRESS / ACCEPTANCE FAILED
Priority: HIGH
Depends on: GEO-001, MATCH-001, GEOM-001
Goal: Build an acquisition-isolated regional production pack using only frozen official RoMa v2 plus regional retrieval, bidirectional geometry, and guarded NCC refinement.

Evidence: `E009S3481` (Apollo 15 S-IVB Impact) was selected by the candidate audit. Twenty official four-band PDS3 observations were acquired and SHA256-validated, with a 12/4/4 full-acquisition TRAIN/VALIDATION/TEST split, 100% canonical map overlap, 162 map-derived GT pair artifacts, and TRAIN-only deterministic retrieval. Measured backplane median ranges are incidence 4.497–83.554°, emission 2.074–27.820°, and phase 1.906–81.933°. The untouched official RoMa v2 precise bidirectional baseline has held-out VRR 0.5 and FAR 0.0 on only three available negative challenges. No NCC result was accepted because all proposals were reverse-inconsistent or degraded geometric residual. The semantic-similarity and shadow-negative categories, accepted NCC controls, per-query demo visuals, and higher positive VRR are missing; the pack must not be claimed as complete.

## RP-001-D1-DEMO

Status: DONE / GATE FAILED
Priority: HIGH
Depends on: RP-001
Goal: Run a transparent, user-authorized experimental regional specialization without changing the official production path.

Evidence: all 66 TRAIN combinations were mask/GT/cycle checked. Nineteen had a >=95% nodata-aware 640px common window and generated 97 unique crops; 47 exclusions are named in the manifest. RAW preprocessing won a TRAIN/VALIDATION-only comparison. D1 trained only `refiners.1`, with all official backbone/matcher/earlier stages frozen, at 1e-6/3e-6/1e-5 and steps 10/25/50/100/250. The validation-selected state is LR 3e-6 / step 10. Paired base-640 metrics: TRAIN PCK@1 `.612917767→.612626400`, VALIDATION PCK@1 `.745905315→.741747784`, validation VRR `3/4→3/4`, TEST PCK@1 `.825662145→.825721564`, and test VRR `4/4→4/4`. There is no meaningful adaptation gain; `PRODUCTION_GATE: FAIL`, `HACKATHON_DEMO_GATE: FAIL`. The ignored demo bundle is only a transparent comparison/provenance artifact.

## MAC-M4-DEMO-001

Status: DONE / HARDWARE VALIDATION PENDING
Priority: HIGH
Depends on: RP-001
Goal: Transfer a fully local, Official-RoMa-v2-only RP-001 presentation demo to a MacBook Air M4.

Evidence: `release/chandrappan_mac_m4_demo/` contains 333 checksummed files (1,585,731,732 bytes): the official checkpoint, a local DINOv3 source cache, 20 processed observations, 12 cached TRAIN descriptors, thumbnails, a local FastAPI presentation app, prepared examples, and setup/start/doctor/preflight scripts. Raw PDS3, GT archive, training data, optimizer/D1 state, CUDA tools, and host caches are excluded. Ubuntu CUDA validated local release loading with an empty host Torch Hub cache, a held-out verified case, a rejected hard negative, and FastAPI integration. MPS execution, M4 timings/memory, CPU fallback, and browser rehearsal require the actual Mac.
