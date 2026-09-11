#!/usr/bin/env python3
"""Run one genuine local RP-001 presentation case with Official RoMa v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.demo_runtime import DemoRuntime


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Offline RP-001 live inference. D1 is never loaded by this command."
    )
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=root,
        help="relocatable Mac release root containing models/ and data/rp001/",
    )
    parser.add_argument(
        "--case",
        choices=("heldout", "difficult_lighting", "hard_negative"),
        default="heldout",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = DemoRuntime(args.demo_root, device=args.device, developer_mode=True)
    runtime.preload()
    result = runtime.run_case(args.case)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
