# Development Handoff

## Branch

`dev`

## Current commit

Pending objective-correction commit

## Goal

Build the verifiable Chandrappan foundation before any model training.

## Current state

Phase 0/GEO-001 and the available real-data geometric baseline are complete. The new training diagnostic is source-faithful at 640x640: it refuses anisotropic resizing, uses physical pixel GT sampled at every refiner stage, applies the RoMa robust loss at strides 4/2/1, and updates only refiners (including refiner-only EMA). The full suite has 44 tests.

## Known blockers

The immutable T0 smoke baseline remains untouched. The historic loss-collapse run is now classified as invalid training evidence because it optimized T0; its runner fails loudly. The non-T0 controlled robust run lowers validation median EPE (4.0009→3.8998 px), raises PCK@3/5, and preserves VRR/FAR (.5/.0), but decreases PCK@1 (.106502→.104818). `FULL TRAINING GATE: FAIL`. Details are in `results/training_objective_fix.{json,md}`. Use `/usr/bin/python3.10`, not the incomplete `/usr/local/bin/python3` build. The ignored vendored checkout has a local checkpoint-path constructor edit; preserve it.

## Next step

Provide LunarMatch-NASA products/manifest, official splits/metadata, and nodata-aware/covisible GT. Increase validation sample size and rerun the strict gate; do not launch full fine-tuning.
