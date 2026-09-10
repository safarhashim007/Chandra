# Known Issues

## KI-001 — RoMa v2 execution unverified on Python 3.14

Severity: HIGH

Affects: baseline matching and training

Reproduction: current host runs Python 3.14.2; upstream lists Python >=3.10 and documents testing on 3.12.

Workaround: create a Python 3.12-compatible isolated environment before downloading a checkpoint or running RoMa.

## KI-002 — No local NASA dataset or product manifest

Severity: HIGH

Affects: catalog build, ground-truth generation, benchmark, and training.

Workaround: the project can validate synthetic geometry and metadata contracts while awaiting an explicit data source.

## KI-003 — `python3` host build lacks SQLite extension

Severity: MEDIUM

Reproduction: `python3 -c 'import sqlite3'` fails with `No module named '_sqlite3'`.

Root cause: the host's Python 3.14 build is missing the optional `_sqlite3` extension.

Impact: invoking project scripts through `python3` breaks the SQLite catalog. The supported local interpreter is `/usr/bin/python` (Python 3.10, SQLite enabled); use `python`, not `python3`, for project commands.
