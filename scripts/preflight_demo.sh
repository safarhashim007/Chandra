#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then echo "FAIL: missing Mac virtual environment" >&2; exit 1; fi
"$PYTHON" "$ROOT/scripts/doctor_mac_demo.py" --demo-root "$ROOT"
test -f "$ROOT/frontend/index.html"
test -f "$ROOT/data/rp001/demo_manifest.json"
test -f "$ROOT/models/romav2_official.pt"
echo "CHANDRAPPAN PRESENTATION READY"
