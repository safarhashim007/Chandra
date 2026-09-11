#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
if [ "$(uname -s)" != "Darwin" ]; then echo "FAIL: this setup script is for macOS" >&2; exit 1; fi
if [ "$(uname -m)" != "arm64" ]; then echo "FAIL: Apple Silicon arm64 is required" >&2; exit 1; fi
PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || { echo "FAIL: Python 3.10+ is required" >&2; exit 1; }
cd "$ROOT"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-demo-mac.txt
.venv/bin/python scripts/doctor_mac_demo.py --demo-root "$ROOT" --skip-inference
echo "Mac dependencies installed. Run ./scripts/start_demo_mac.sh"
