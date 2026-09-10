#!/usr/bin/env bash
set -euo pipefail
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
mkdir -p data/{raw,processed,tiles,manifests,indexes,ground_truth,cache} artifacts logs/debug benchmarks checkpoints
.venv/bin/python scripts/system_check.py
.venv/bin/python -m pytest
