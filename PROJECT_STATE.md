# Chandrappan Project State

Last updated: 2026-09-10 Asia/Kolkata

Current version: pre-v0.1

Current stable commit: `ac651be` (workflow baseline; lunar primitives pending checkpoint)

Integration branch: `dev`

Primary production matcher: RoMa v2 (not yet installed/benchmarked)

Experimental matcher: LunarRoMa (not started)

## Current phase

Phase 0 / repository workflow and environment health check. The public RoMa v2 source checkout is at `vendor/romav2`, commit `95c9968145c8906b7b59383258e9f73b02853d89`; no checkpoint has been downloaded and no training has occurred. Supported local interpreter is `/usr/bin/python` (Python 3.10).

## Completed

- [x] Empty workspace initialized as a `dev` Git repository.
- [x] Environment inspected: Linux x86_64, Python 3.14.2, NVIDIA RTX 4000 SFF Ada (20,475 MiB), 62 GiB RAM, 86 GiB free disk.
- [x] Official RoMa v2 source cloned for inspection.

## In progress

- [x] Repository-management infrastructure and Phase 0 tools.
- [x] Moon-specific longitude conventions, affine pixel/world transforms, metadata contract, SQLite/RTree catalog implementation, and synthetic roundtrip tests.

## Blockers

- No NASA imagery, catalog manifest, or official RoMa checkpoint has been supplied or downloaded.
- `/usr/local/bin/python3` is Python 3.14 and lacks `_sqlite3`; project commands must use `/usr/bin/python` (Python 3.10) until a coherent modern environment is provisioned.

## Last commands

`python3 scripts/system_check.py`; `python3 scripts/doctor.py`; `python3 scripts/inspect_romav2.py`; `PYTHONPATH=. pytest -q`.

## Next exact task

Provide a NASA LROC product/metadata manifest, then validate the metadata adapter against a real map-projected product before catalog ingestion.
