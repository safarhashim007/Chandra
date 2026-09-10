# Chandrappan Project State

Last updated: 2026-09-10 Asia/Kolkata

Current version: pre-v0.1

Current stable commit: pending this session's verified ingestion/training-readiness changes

Integration branch: `dev`

Primary production matcher: RoMa v2 (official checkpoint loaded and GPU smoke-tested)

Experimental matcher: LunarRoMa (not started)

## Current phase

Phase 1 / GEO-001 is complete for the available real products. MATCH-001 is in progress: the public RoMa v2 source checkout is at `vendor/romav2`, commit `95c9968145c8906b7b59383258e9f73b02853d89`, and the ignored official v2.0.1 checkpoint is available locally. Supported local interpreter is `/usr/bin/python` (Python 3.10).

## Completed

- [x] Empty workspace initialized as a `dev` Git repository.
- [x] Environment inspected: Linux x86_64, Python 3.14.2, NVIDIA RTX 4000 SFF Ada (20,475 MiB), 62 GiB RAM, 86 GiB free disk.
- [x] Official RoMa v2 source cloned for inspection.
- [x] Downloaded an ignored public LROC WAC map-projected GeoTIFF/XML fixture and independently verified its GeoTIFF affine tags and Moon center.

## In progress

- [x] Repository-management infrastructure and Phase 0 tools.
- [x] Moon-specific longitude conventions, affine pixel/world transforms, metadata contract, SQLite/RTree catalog implementation, and synthetic roundtrip tests.
- [x] Real-product tag-level affine roundtrip checked for `WAC_TIO2_E350N0450` with maximum error `2.6e-13 px`.
- [x] Rasterio/pyproj LROC adapter validated against the WAC fixture; all project tests pass with the real-product test enabled.
- [x] Two real map-projected NAC photometric products were ingested into an ignored SQLite/RTree catalog and their footprints were verified to overlap.
- [x] Official RoMa v2.0.1 checkpoint loaded; turbo and base CUDA inference smoke tests returned finite normalized warps and confidence tensors.
- [x] Separate gradient-enabled training forward implemented; a CUDA gradient smoke test reached the refiner parameters while frozen descriptor/matcher parameters stayed without gradients.
- [x] Pixel-centre GT and official RoMa coordinate semantics are covered by regression tests.

## Blockers

- A proper LunarMatch-NASA training manifest, geographic train/validation/test split, and held-out benchmark are not available.
- The one-batch refiner-only diagnostic on the available NAC pair reduced robust training loss but worsened PCK/EPE; the mandatory overfit gate therefore remains failed.
- `/usr/local/bin/python3` is Python 3.14 and lacks `_sqlite3`; project commands must use `/usr/bin/python` (Python 3.10).
- The vendored RoMa checkout has an existing local constructor/checkpoint-path modification; it is ignored and must not be overwritten or committed without explicit ownership review.

## Last commands

`make format`; `make lint`; `make test`; `make test-geo`; `make test-training` — all passed using `/usr/bin/python` 3.10. Current suite: 13 tests. Official CUDA smoke inference and the gradient smoke both passed; the one-batch overfit gate did not.

## Next exact task

Acquire or provide the LunarMatch-NASA products and manifest with geographic splits and photometric metadata; then establish an immutable T0 benchmark, make the one-batch test pass on a valid training sample, and only then start T1 refiner fine-tuning.
