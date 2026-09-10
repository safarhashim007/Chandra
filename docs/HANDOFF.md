# Development Handoff

## Branch

`dev`

## Current commit

`6d71896` (verified ingestion and RoMa training-readiness changes)

## Goal

Build the verifiable Chandrappan foundation before any model training.

## Current state

Phase 0 is complete. GEO-001 is complete for the available real products: the WAC fixture passes the rasterio-backed adapter and a real NAC photometric pair is cataloged with overlapping footprints. Public RoMa v2 source is inspected from an ignored vendor checkout; the official v2.0.1 checkpoint loads and CUDA inference smoke tests pass. A separate gradient-enabled training forward and centralized freezing helpers exist; 13 project tests pass.

## Known blockers

No LunarMatch-NASA training manifest, geographic split, or held-out T0 benchmark is available. The mandatory one-batch refiner-only diagnostic on the available NAC crop lowers robust loss but worsens pixel metrics, so TRAIN-001 remains blocked. Use `/usr/bin/python` (Python 3.10), not the incomplete `/usr/local/bin/python3` build. The ignored vendored RoMa checkout contains an existing local checkpoint-path constructor edit; preserve it unless explicitly reviewed.

## Next step

Provide the LunarMatch-NASA products/manifest and geographic train/validation/test split; then build immutable T0, diagnose the one-batch gate, and only after it passes launch staged T1 fine-tuning.
