# Development Handoff

## Branch

`dev`

## Current commit

`6358c8d` (last substantive fixture/state commit)

## Goal

Build the verifiable Chandrappan foundation before any model training.

## Current state

The workspace began empty. Phase 0 is complete and synthetic lunar geometry tests pass. A public LROC WAC map-projected GeoTIFF/XML fixture (`WAC_TIO2_E350N0450`, MD5 `7774860882ef0330cf6afdacbca2b731`) was downloaded under ignored `data/raw/lroc_fixture/`. Its GeoTIFF tags and PDS XML were checked directly; the derived affine pixel/world roundtrip error was `2.6e-13 px`, with the expected equirectangular Moon radius and 35°N/45°E center. Public RoMa v2 source is inspected from an ignored vendor checkout; no checkpoint/model run has occurred. Initial code implements lunar conventions, affine pixel/world roundtrips, a metadata contract, an SQLite/RTree catalog, pixel-space GT generation, RoMa coordinate conversions, and mask-safe loss primitives.

## Known blockers

The declared geospatial runtime packages (`rasterio`, `pyproj`, `shapely`) are not installed in `/usr/bin/python`; the real-product check was therefore tag-level via installed `tifffile`, not a project ingestion test. No official RoMa checkpoint is present. Use `/usr/bin/python` (Python 3.10) for project commands, not the incomplete `/usr/local/bin/python3` build.

## Next step

Provision the declared geospatial runtime, implement the rasterio-backed LROC metadata adapter, and add a real-product metadata/roundtrip regression test using the ignored fixture. Do not ingest it into the catalog until that test passes.
