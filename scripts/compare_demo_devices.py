#!/usr/bin/env python3
"""Record or compare local parity decisions for the fixed held-out demo case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chandrappan.demo_runtime import DemoRuntime


def compact(result: dict[str, object]) -> dict[str, object]:
    verification = result.get("verification", {})
    selected = verification.get("selected") if isinstance(verification, dict) else None
    return {
        "verdict": result.get("verdict"),
        "selected_reference": result.get("selected_reference"),
        "geometric_match_count": result.get("geometric_match_count"),
        "inliers": selected.get("inliers") if isinstance(selected, dict) else None,
        "transform": selected.get("matrix") if isinstance(selected, dict) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = DemoRuntime(args.demo_root, device=args.device, developer_mode=True)
    runtime.preload()
    current = compact(runtime.run_case("heldout"))
    payload: dict[str, object] = {"device": args.device, "heldout": current}
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())["heldout"]
        payload["baseline"] = baseline
        payload["same_verdict"] = current["verdict"] == baseline["verdict"]
        payload["same_reference"] = current["selected_reference"] == baseline["selected_reference"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
