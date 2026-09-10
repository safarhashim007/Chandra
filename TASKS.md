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
