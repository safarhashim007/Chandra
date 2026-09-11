#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then echo "FAIL: run ./scripts/setup_mac_m4.sh first" >&2; exit 1; fi
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then echo "FAIL: Mac M4 presentation launch requires macOS arm64" >&2; exit 1; fi
cd "$ROOT"
mkdir -p artifacts/live
CHANDRAPPAN_DEMO_ROOT="$ROOT" CHANDRAPPAN_DEVICE=auto "$PYTHON" -m uvicorn api.app:app --host 127.0.0.1 --port 8000 > artifacts/live/server.log 2>&1 &
PID=$!
echo "$PID" > artifacts/live/server.pid
COUNT=0
until curl -fsS http://127.0.0.1:8000/api/demo/health >/dev/null 2>&1; do
  COUNT=$((COUNT + 1))
  if [ "$COUNT" -ge 90 ]; then echo "FAIL: model preload did not complete; see artifacts/live/server.log" >&2; exit 1; fi
  sleep 1
done
echo "CHANDRAPPAN MODEL READY"
echo "Open: http://127.0.0.1:8000"
open http://127.0.0.1:8000 >/dev/null 2>&1 || true
