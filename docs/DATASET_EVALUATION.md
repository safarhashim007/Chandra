# Dataset and evaluation milestone

## Canonical manifest

`chandrappan.data.manifest` writes Parquet rows with explicit nulls for unavailable
acquisition and photometric fields. It validates paths, IDs, dimensions, GSD, numeric
bounds, and non-empty valid WKT footprints. LROC rows are created without inventing
incidence, emission, phase, or acquisition values.

Build the current real-data smoke manifest with:

```bash
/usr/bin/python scripts/build_dataset.py \
  data/raw/lroc_fixture/WAC_TIO2_E350N0450.TIF \
  data/raw/lroc_pair/NAC_PHO_E018N3346_M107042466L.TIF \
  data/raw/lroc_pair/NAC_PHO_E018N3346_M107042466R.TIF \
  --output-root data/splits --seed 1
```

The available smoke data has one WAC region and one two-image NAC region. Therefore
the deterministic split has one training image, an empty validation split, and two
test images. This validates mechanics but is not sufficient for threshold selection
or model training.

## Geographic isolation and pairs

`geographic_split()` assigns complete `region_id` groups to one split. `validate_no_leakage()`
rejects repeated image IDs, repeated regions, and forbidden cross-split footprint overlap.
`generate_pairs()` uses footprint intersection and rejects negligible overlap; it records
overlap ratio, scale ratio, and nullable acquisition/photometric differences.

## Project-defined T0

`benchmarks/T0_v1.parquet` is a frozen one-pair smoke benchmark created from the available
NAC pair. Its checksum is `benchmarks/T0_v1.sha256`. This is explicitly **project-defined**,
not the official LunarMatch-NASA benchmark. Evaluation verifies the checksum before running.
Training preflight rejects both the T0 path and byte-identical copies.

The baseline command is:

```bash
PYTHONPATH=vendor/romav2/src /usr/bin/python scripts/evaluate_t0.py
```

The current baseline is in `results/T0/`. Correspondence metrics are available; VRR/FAR,
geometric inliers, reprojection error, and spatial coverage remain null until a geometric
verifier and negative set are available.

## Training quality gate

`chandrappan.evaluation.quality_gate` compares the same validation metrics before and after
short refiner-only training. It requires no median-EPE/PCK@1/VRR regression, no FAR increase,
and an improvement in PCK@3 or PCK@5. `chandrappan.training.preflight.assert_training_allowed`
must be called before any future full-training entry point.

The current gate is FAIL. The available smoke pair has no validation split, and its controlled
one-batch diagnostic worsens median EPE/PCK@1 under the current objective.
