# Chandrappan Tasks

## REPO-001

Status: IN_PROGRESS  
Priority: HIGH  
Goal: Establish reproducible repository workflow, health checks, documentation, and CPU-safe tests.

Acceptance: state/handoff files, doctor/system checks, format/lint/test commands, and initial commit.

## GEO-001

Status: READY  
Priority: HIGH  
Depends on: REPO-001  
Goal: Implement Moon-specific CRS, pixel/world transforms, footprint metadata, catalog, and roundtrip tests.

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
