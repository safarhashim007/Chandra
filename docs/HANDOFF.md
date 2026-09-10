# Development Handoff

## Branch

`dev`

## Current commit

`3c757da` (geometric verification, real validation, and diagnostics)

## Goal

Build the verifiable Chandrappan foundation before any model training.

## Current state

Phase 0 and GEO-001 are complete for the available real products. The canonical Parquet manifest, geographic split/leakage checks, pair generation, frozen project-defined T0, untouched CUDA baseline, VRR/FAR protocol, and validation gate are implemented. Public RoMa v2 source is inspected from an ignored vendor checkout; the official v2.0.1 checkpoint loads and CUDA inference smoke tests pass. A separate gradient-enabled training forward and centralized freezing helpers exist; 23 project tests pass.

## Known blockers

The immutable T0 smoke baseline has one pair and remains separate from threshold selection. A genuine expanded LROC corpus now has 20 images across 7 regions, with 2 validation positives and 4 validation negatives. The untouched RoMa baseline is VRR 1.0 and FAR 0.0 under validation-selected thresholds; diagnostic images and outlier/confidence analyses are in `results/validation/`. The one-batch refiner-only diagnostic lowers robust loss but fails the quality gate; TRAIN-001 remains blocked. Use `/usr/bin/python` (Python 3.10), not the incomplete `/usr/local/bin/python3` build. The ignored vendored RoMa checkout contains an existing local checkpoint-path constructor edit; preserve it unless explicitly reviewed.

## Next step

Provide the LunarMatch-NASA products/manifest and official geographic split/photometric metadata; resolve the training objective collapse, and only after the validation gate passes consider staged T1 fine-tuning.
