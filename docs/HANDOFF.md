# Development Handoff

## Branch

`dev`

## Current commit

`46ea2cc` (dataset, T0, gate, and regression milestone)

## Goal

Build the verifiable Chandrappan foundation before any model training.

## Current state

Phase 0 and GEO-001 are complete for the available real products. The canonical Parquet manifest, geographic split/leakage checks, pair generation, frozen project-defined T0, untouched CUDA baseline, VRR/FAR protocol, and validation gate are implemented. Public RoMa v2 source is inspected from an ignored vendor checkout; the official v2.0.1 checkpoint loads and CUDA inference smoke tests pass. A separate gradient-enabled training forward and centralized freezing helpers exist; 23 project tests pass.

## Known blockers

The smoke dataset has 1 train image, 0 validation images, and 2 test images, so it cannot support training or threshold selection. The T0 smoke baseline has one pair and no geometric/VRR/FAR metrics. The one-batch refiner-only diagnostic lowers robust loss but fails the quality gate; TRAIN-001 remains blocked. Use `/usr/bin/python` (Python 3.10), not the incomplete `/usr/local/bin/python3` build. The ignored vendored RoMa checkout contains an existing local checkpoint-path constructor edit; preserve it unless explicitly reviewed.

## Next step

Provide the LunarMatch-NASA products/manifest and official geographic split/photometric metadata; add real validation and negative regions, implement geometric verification, and only after the validation gate passes launch staged T1 fine-tuning.
