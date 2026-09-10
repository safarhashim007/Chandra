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

Status: IN_PROGRESS
Priority: HIGH
Depends on: GEO-001
Goal: Establish official RoMa v2 baseline adapter and inspect actual tensor conventions.

Evidence: official v2.0.1 checkpoint loads on CUDA; turbo and base inference smoke tests return finite outputs. A held-out T0 benchmark is still unavailable.

## TRAIN-001

Status: BLOCKED
Priority: HIGH
Depends on: MATCH-001, LunarMatch-NASA data, T0 benchmark, one-batch overfit
Goal: Add a validated, source-compatible LunarRoMa training path without modifying official inference.
