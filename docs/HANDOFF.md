# Development Handoff

## Branch

`dev`

## Current commit

`feb7b8e`

## Goal

Build the verifiable Chandrappan foundation before any model training.

## Current state

The workspace began empty. Phase 0 is complete and synthetic lunar geometry tests pass. Public RoMa v2 source is inspected from an ignored vendor checkout; no checkpoint/model run has occurred. Initial code implements lunar conventions, affine pixel/world roundtrips, a metadata contract, an SQLite/RTree catalog, pixel-space GT generation, RoMa coordinate conversions, and mask-safe loss primitives.

## Known blockers

No NASA data or checkpoint is present. Use `/usr/bin/python` (Python 3.10) for project commands, not the incomplete `/usr/local/bin/python3` build.

## Next step

Run `make doctor` and `make test`. Then supply or locate a small LROC map-projected product plus metadata sidecar and add a real-product metadata/roundtrip regression test.
