#!/usr/bin/env python3
"""Run the required MPS resolution benchmark on the transferred Mac."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from chandrappan.demo_runtime import DemoRuntime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case", default="heldout")
    parser.add_argument("--output", type=Path, default=Path("artifacts/resolution_benchmark.json"))
    args = parser.parse_args()
    runtime = DemoRuntime(args.demo_root, device="mps", developer_mode=True)
    runtime.preload()
    rows = []
    for resolution in (448, 512, 640):
        runtime.config["live_demo_resolution"] = resolution
        started = time.perf_counter()
        result = runtime.run_case(args.case)
        allocated = (
            int(__import__("torch").mps.current_allocated_memory())
            if hasattr(__import__("torch").mps, "current_allocated_memory")
            else None
        )
        rows.append(
            {
                "resolution": resolution,
                "verdict": result["verdict"],
                "selected_reference": result["selected_reference"],
                "total_seconds": time.perf_counter() - started,
                "timing": result["timing"],
                "mps_allocated_bytes_after_run": allocated,
            }
        )
    output = args.demo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"device": "mps", "rows": rows}, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
