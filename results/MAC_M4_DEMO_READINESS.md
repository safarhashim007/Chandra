# Chandrappan Mac M4 demo readiness

Date: 2026-09-11

## Transfer

- Release bundle: `release/chandrappan_mac_m4_demo/`
- Transfer manifest: `release/chandrappan_mac_m4_demo/TRANSFER_MANIFEST.md`
- Bundle size: `1,585,731,732` bytes.
- Bundle file count: `333`.
- Model files: one Official RoMa v2 checkpoint plus a pinned local DINOv3 source cache.
- RP-001 assets: 20 processed science rasters, 20 descriptors, 20 thumbnails, local metadata,
  three prepared cases, and one compact held-out parity fixture.
- Excluded: raw PDS3 corpus, full GT archive, D1 checkpoint, training crops, optimizer state,
  Ubuntu virtual environment, CUDA/NVIDIA packages, compiler and package caches.

## Mac runtime

Target: Apple Silicon `arm64`, MacBook Air M4, 16 GB unified memory, Apple MPS.

- Profile: `configs/demo_mac_m4.yaml`.
- Mode: inference only; batch size 1; Official RoMa v2 `ACTIVE · FROZEN`.
- MPS priority: MPS, then CPU. macOS never attempts CUDA.
- LIVE_DEMO_RESOLUTION: `640`. It preserves the base-640 scientific protocol used for the stored
  paired metrics. `scripts/benchmark_demo_resolution.py` measures 448/512/640 on the M4 before
  any timing or memory claim is made.
- Model, descriptors, metadata, and thumbnails preload once. The server exposes `MODEL READY`
  only after warmup completes.
- Known live cases: held-out validation, held-out TEST different-lighting case, and a real other-
  region hard negative. Runtime output uses generated artifacts; no static result is substituted.

## Offline readiness

Internet required after setup: `NO`.

The release has local weights, pinned DINOv3 source, processed local query/reference data,
descriptors, thumbnails, frontend, and static fallback. The Mac setup step may need package access
once to install Apple-Silicon Python wheels; the presentation path does not.

## Visualizations

Live frontend confirms:

- actual reference-search candidate cards and selected rank;
- Canvas correspondence lines using spatially distributed verified inliers;
- overlay, draggable wipe, blink/pause, split, registered difference, and provenance;
- presentation mode for a 13-inch screen;
- scientific-details drawer and D1 provenance panel;
- a non-error hard-negative `REJECTED` outcome.

Static fallback: `demo/RP-001/final/index.html`. It is visibly marked `PRECOMPUTED DEMO` and does
not claim live inference.

## Scientific presentation metrics

Stored Official RoMa v2 TEST values, paired base-640 protocol:

| Metric | Exact | Presentation |
|---|---:|---:|
| PCK@0.5 | 0.799365472 | 79.9% |
| PCK@1 | 0.825662145 | 82.6% |
| Median EPE | 0.347608089 px | 0.35 px |
| VRR | 4/4 = 1.0 | 100% · 4/4 |

Supporting corpus values: 20 NASA LROC observations, 12 TRAIN acquisitions, 19 scientifically
valid pairs, and 97 dense-GT crops. D1 is `EXPERIMENTAL · NOT PROMOTED`; it is never the live
matcher.

## Verification performed on Ubuntu

- Bundle builder completed and wrote checksums.
- Empty-host-cache offline test passed: the live release used its bundled DINOv3 cache.
- Official CUDA live held-out case: `VERIFIED`, selected rank #1, 4,000 geometric matches and
  4,000 inliers; warm result latency was 0.7023 s on this Ubuntu GPU. This timing does not apply
  to the M4.
- Official CUDA hard negative: `REJECTED` after 2,451 attempted geometric matches.
- FastAPI integration: health, summary, and difficult-lighting case returned HTTP 200 with
  `MODEL READY` and `VERIFIED`.
- Full pytest suite: `65 passed` (15 existing Matplotlib warnings).
- Formatting and targeted lint for every Mac-demo file: PASS.
- `make lint` remains blocked by the pre-existing untracked
  `scripts/build_rp001_handpicked_pack.py` (22 unrelated findings). It is not part of the release
  bundle and was preserved without modification.

## Remaining Mac-only validation

- MPS model load and full live path.
- MPS 448/512/640 latency and unified-memory benchmark.
- CUDA→MPS selected-reference/verdict/transform parity.
- CPU fallback run.
- Browser layout and full-screen rehearsal on the 13-inch MacBook Air.

`MAC M4 OFFLINE DEMO BUILD: PASS`

`MAC M4 HARDWARE VALIDATION: PENDING`
