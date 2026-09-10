# Chandrappan Project State

Last updated: 2026-09-11 Asia/Kolkata

Current version: pre-v0.1

Current stable commit: `3d6928c` (final dataset, T0, gate, and regression handoff)

Integration branch: `dev`

Primary production matcher: RoMa v2 (official checkpoint loaded and GPU smoke-tested)

Experimental matcher: LunarRoMa (not started)

## Current phase

Phase 2 dataset/evaluation foundation is complete for the available real products. MATCH-001 and TRAIN-001 remain blocked for production claims: the public RoMa v2 source checkout is at `vendor/romav2`, commit `95c9968145c8906b7b59383258e9f73b02853d89`, and the ignored official v2.0.1 checkpoint is available locally. Supported local interpreter is `/usr/bin/python` (Python 3.10).

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
- [x] Canonical Parquet image manifest, deterministic geographic splits, leakage checks, and footprint-overlap pair generation implemented.
- [x] Project-defined one-pair T0 benchmark frozen and SHA256 checked; untouched RoMa v2 baseline recorded on CUDA.
- [x] VRR/FAR acceptance definitions and a hard validation quality gate implemented; T0 is rejected from training by path or checksum.
- [x] Known loss/EPE/PCK regression reproduced with beta=1.0 and an objective-scale ablation recorded with beta=0.01.

## Blockers

- A proper LunarMatch-NASA training manifest, geographic train/validation/test split, and held-out benchmark are not available.
- The project-defined smoke split has 1 train image, 0 validation images, and 2 test images; it is not sufficient for threshold selection or full training.
- The one-batch refiner-only diagnostic on the available NAC pair reduced robust training loss but worsened PCK@1/median EPE; the mandatory quality gate therefore remains failed.
- The T0 smoke baseline has no geometric inliers, VRR, FAR, or coverage because no geometric verifier/negative set exists yet.
- `/usr/local/bin/python3` is Python 3.14 and lacks `_sqlite3`; project commands must use `/usr/bin/python` (Python 3.10).
- The vendored RoMa checkout has an existing local constructor/checkpoint-path modification; it is ignored and must not be overwritten or committed without explicit ownership review.

## Last commands

`make format`; `make lint`; `make test`; `make test-geo`; `make test-matching`; `make test-training` — all passed using `/usr/bin/python` 3.10. Current suite: 23 tests. Dataset build, T0 evaluation, and regression reproduction ran on CUDA; full fine-tuning was not launched.

## Next exact task

Acquire or provide the LunarMatch-NASA products and official geographic split/metadata; add a real validation region and geometric verifier/negative set, then make the validation quality gate pass before any T1 fine-tuning.
