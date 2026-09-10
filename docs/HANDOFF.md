# Development Handoff

## Branch

`dev`

## Goal

Build the verifiable Chandrappan foundation before any model training.

## Current state

The workspace began empty. Repository workflow initialization and Phase 0 are in progress. Public RoMa v2 source is inspected from an ignored vendor checkout, but no checkpoint/model run has occurred.

## Known blockers

No NASA data or checkpoint is present. Python 3.14 compatibility with upstream RoMa v2 remains unverified.

## Next step

Run `python3 scripts/doctor.py` and `make test`; then continue GEO-001 only after the persistent state files are current.
