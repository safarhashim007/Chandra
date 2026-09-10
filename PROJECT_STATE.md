# Chandrappan Project State

Last updated: 2026-09-11 Asia/Kolkata

Current version: pre-v0.1

Current stable commit: pending objective-correction commit

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
- [x] Training objective audit completed: beta=1.0 is 320 px/axis at 640, while beta=.01 is 3.2 px/axis; coordinate and displacement round trips pass on non-square grids.
- [x] Corrected refiner-only path uses source RoMa robust EPE loss at strides 4/2/1, stage-wise GT sampling, refiner-only EMA, frozen-parameter checks, T0-pair rejection, and a source-faithful 640x640 input contract.
- [x] Bounded CUDA LR ablation completed on a geographically TRAIN-only LROC pair and evaluated on held-out validation. The conservative run improves median EPE/PCK@3/PCK@5 but slightly decreases PCK@1, so the strict training gate remains failed.
- [x] Expanded genuine LROC NAC corpus built: 20 images across 7 geographic regions, with leakage-safe train/validation/test manifests and positive/negative pair manifests.
- [x] Untouched RoMa v2 validation baseline recorded: 2 positive pairs, 4 negatives, VRR 1.0, FAR 0.0 under the selected project configuration; confidence and outlier distributions recorded.

## Blockers

- A proper LunarMatch-NASA training manifest, geographic train/validation/test split, and held-out benchmark are not available.
- The immutable project-defined T0 remains a one-pair smoke benchmark; the expanded LROC corpus is a project validation corpus, not official LunarMatch-NASA.
- Full training remains blocked: the best controlled run has validation PCK@1 `.104818` versus untouched `.106502`, despite improving median EPE `4.0009→3.8998 px`, PCK@3 `.3479→.3680`, PCK@5 `.5226→.5253`, and retaining VRR/FAR `.5/.0`.
- Validation geometry is available, but the current T0 result remains positive-only and is not used for threshold selection or FAR.
- `/usr/local/bin/python3` is Python 3.14 and lacks `_sqlite3`; project commands must use `/usr/bin/python` (Python 3.10).
- The vendored RoMa checkout has an existing local constructor/checkpoint-path modification; it is ignored and must not be overwritten or committed without explicit ownership review.

## Last commands

`make format`; `make lint`; `make test`; `make test-geo`; `make test-matching`; `make test-training`; `make doctor` — all passed using `/usr/bin/python3.10`. Current suite: 44 tests. Corrected CUDA objective and held-out validation ablations ran without full fine-tuning.

## Next exact task

Acquire LunarMatch-NASA products, official splits/metadata, and nodata-aware/covisible supervision; expand the real validation corpus before rerunning the strict gate.

## Latest curation run

`6d894b6` created `runs/lroc_curated_v1` from the existing genuine 20-product NAC_PHO TIFF/XML corpus. Twelve TRAIN positive pairs passed 640px overlap-window dense-GT/mask/cycle validation; three pairs from one available TRAIN region were selected and SHA256-frozen. The archive expansion target was not met, and one raster read failure was classified as a rejection. Full fine-tuning was not launched.
