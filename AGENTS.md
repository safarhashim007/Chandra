# Chandrappan Agent Instructions

## Mission

Chandrappan is a NASA lunar-image localization, retrieval, dense-registration and verification system. The production matcher is RoMa v2; LunarRoMa is an experimental, optional adaptation. Production must retain the untouched RoMa v2 fallback.

## Golden rules

1. Do not fabricate registrations, metrics, model improvements, or successful outputs.
2. Use a Moon-specific CRS internally: east-positive longitude, planetocentric latitude, mean radius 1,737,400 m where a spherical lunar CRS is required.
3. Preserve official RoMa v2 inference. Training work must use a separate gradient-enabled path.
4. Do not change scientific thresholds to make tests pass. Registration is never accepted on RMSE alone.
5. Do not swallow exceptions. Capture, classify, minimally reproduce, fix the earliest cause, and add a regression test.
6. Do not commit raw NASA data, model weights, caches, credentials, or generated artifacts.
7. Read `PROJECT_STATE.md`, `TASKS.md`, `KNOWN_ISSUES.md`, and `DECISIONS.md` before substantive edits; update state and handoff before stopping.

## Required checks

Before editing: `git status`, `git branch --show-current`, and `git log --oneline -10`.

Before a commit: `make format`, `make lint`, `make test`, and the relevant component checks. Never mark work complete without executed tests and updated persistent documentation.

## Current phase guard

Implement phases in order. Do not launch LunarRoMa training until RoMa internals, coordinate/GT/loss tests, gradient checks, one-batch overfit, and the immutable T0 baseline have passed.
