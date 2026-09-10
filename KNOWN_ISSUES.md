# Known Issues

## KI-001 — Incomplete Python 3.14 runtime remains unsupported

Severity: HIGH

Affects: baseline matching and training

Reproduction: `/usr/local/bin/python3` is Python 3.14.2 and lacks `_sqlite3`; official RoMa execution was verified with `/usr/bin/python` 3.10 instead.

Workaround: use `/usr/bin/python` for project commands, or provision a coherent Python 3.12+ environment with SQLite and the declared dependencies.

## KI-002 — No LunarMatch-NASA training dataset or geographic split

Severity: HIGH

Affects: training-pair generation, held-out T0 benchmark, and fine-tuning.

Workaround: the project can validate the geometry contract and use the small ignored NAC pair for smoke tests, but it cannot claim a training or held-out benchmark result.

## KI-003 — `python3` host build lacks SQLite extension

Severity: MEDIUM

Reproduction: `python3 -c 'import sqlite3'` fails with `No module named '_sqlite3'`.

Root cause: the host's Python 3.14 build is missing the optional `_sqlite3` extension.

Impact: invoking project scripts through `python3` breaks the SQLite catalog. The supported local interpreter is `/usr/bin/python` (Python 3.10, SQLite enabled); use `python`, not `python3`, for project commands.

## KI-004 — Declared geospatial runtime packages were previously absent

Severity: HIGH

Affects: real LROC ingestion and raster-backed metadata validation.

Resolution: `/usr/bin/python` now imports rasterio 1.4.4, pyproj 3.7.1, and shapely 2.1.2; the WAC ingestion regression passes.

Current status: resolved in the current workspace; keep the adapter's clear missing-dependency error for clean environments.

## KI-005 — One-batch T1 overfit gate is not passed

Severity: HIGH

Affects: all LunarRoMa fine-tuning stages.

Evidence: on the available 320x320 NAC crop, refiner-only CUDA optimization reduced robust warp loss from `0.04048` to `0.00921`, but median EPE increased from `1.58 px` to `11.61 px` and PCK@1 fell from `0.470` to `0.0002` after 50 steps.

Required action: diagnose the training objective/data sample with a valid LunarMatch-NASA batch, then rerun the one-batch gate. Do not launch full training while this issue remains.

## KI-006 — Official LunarMatch-NASA data and benchmark remain unavailable

Severity: HIGH

Affects: official training, official split compliance, and final benchmark claims.

Current status: the project-defined T0 remains one real NAC pair and is immutable. A separate real LROC validation corpus now has 20 images, two positive pairs, and four negatives; its geometric baseline has measurable VRR/FAR. This is not the official LunarMatch-NASA benchmark.

Required action: provide the official LunarMatch-NASA products, split rules, complete metadata, and official acceptance protocol before making official benchmark or training claims.
