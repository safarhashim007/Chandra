# Chandrappan Project State

Last updated: 2026-09-10 Asia/Kolkata

Current version: pre-v0.1

Current stable commit: `6358c8d` (verified LROC fixture provenance and persistent state)

Integration branch: `dev`

Primary production matcher: RoMa v2 (not yet installed/benchmarked)

Experimental matcher: LunarRoMa (not started)

## Current phase

Phase 1 / GEO-001 real-product metadata validation. The public RoMa v2 source checkout is at `vendor/romav2`, commit `95c9968145c8906b7b59383258e9f73b02853d89`; no checkpoint has been downloaded and no training has occurred. Supported local interpreter is `/usr/bin/python` (Python 3.10).

## Completed

- [x] Empty workspace initialized as a `dev` Git repository.
- [x] Environment inspected: Linux x86_64, Python 3.14.2, NVIDIA RTX 4000 SFF Ada (20,475 MiB), 62 GiB RAM, 86 GiB free disk.
- [x] Official RoMa v2 source cloned for inspection.
- [x] Downloaded an ignored public LROC WAC map-projected GeoTIFF/XML fixture and independently verified its GeoTIFF affine tags and Moon center.

## In progress

- [x] Repository-management infrastructure and Phase 0 tools.
- [x] Moon-specific longitude conventions, affine pixel/world transforms, metadata contract, SQLite/RTree catalog implementation, and synthetic roundtrip tests.
- [x] Real-product tag-level affine roundtrip checked for `WAC_TIO2_E350N0450` with maximum error `2.6e-13 px`.

## Blockers

- No official RoMa checkpoint has been supplied or downloaded. The downloaded LROC fixture is represented by `data/manifests/lroc_fixture.json` but is not yet ingested into the catalog.
- `/usr/local/bin/python3` is Python 3.14 and lacks `_sqlite3`; project commands must use `/usr/bin/python` (Python 3.10) until a coherent modern environment is provisioned.
- The supported interpreter does not currently have the declared geospatial runtime packages (`rasterio`, `pyproj`, `shapely`); the real-product check used installed `tifffile` only and has not yet exercised the project ingestion adapter.

## Last commands

`make doctor`; `make test`; `make lint` — all passed (8 tests) using `/usr/bin/python` 3.10. A direct real-product tag/roundtrip check also passed; no rasterio-backed ingestion test has run.

## Next exact task

Provision the declared geospatial runtime, implement the rasterio-backed LROC metadata adapter, and turn the downloaded fixture plus its MD5 (`7774860882ef0330cf6afdacbca2b731`) into a real-product regression test before catalog ingestion.
