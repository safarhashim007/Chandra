# Chandrappan Tasks

## REPO-001

Status: DONE  
Priority: HIGH  
Goal: Establish reproducible repository workflow, health checks, documentation, and CPU-safe tests.

Acceptance: state/handoff files, doctor/system checks, format/lint/test commands, and initial commit.

## GEO-001

Status: IN_PROGRESS  
Priority: HIGH  
Depends on: REPO-001  
Goal: Validate the implemented Moon-specific CRS, pixel/world transforms, footprint metadata, and catalog against real LROC map-projected products.

Evidence so far: the ignored `WAC_TIO2_E350N0450` LROC WAC GeoTIFF/XML fixture is recorded in `data/manifests/lroc_fixture.json`, has the expected Moon equirectangular tags, and has a direct affine roundtrip error of `2.6e-13 px`. The project ingestion adapter and catalog insertion remain unverified because the declared geospatial runtime is not installed.

## MATCH-001

Status: TODO  
Priority: HIGH  
Depends on: GEO-001  
Goal: Establish official RoMa v2 baseline adapter and inspect actual tensor conventions.

## TRAIN-001

Status: BLOCKED  
Priority: HIGH  
Depends on: MATCH-001, LunarMatch data, T0 benchmark  
Goal: Add a validated, source-compatible LunarRoMa training path without modifying official inference.
