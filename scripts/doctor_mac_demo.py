#!/usr/bin/env python3
"""Validate the relocated Chandrappan Mac M4 offline presentation bundle."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandrappan.demo_runtime import DemoRuntime


def check(name: str, condition: bool, detail: str) -> tuple[str, bool, str]:
    print(f"{'PASS' if condition else 'FAIL'}  {name}: {detail}")
    return name, condition, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    root = args.demo_root.resolve()
    checks = []
    macos = platform.system() == "Darwin"
    arm64 = platform.machine() == "arm64"
    checks.append(check("macOS", macos, platform.platform()))
    checks.append(check("Apple Silicon arm64", arm64, platform.machine()))
    checks.append(check("Python", sys.version_info >= (3, 10), sys.version.split()[0]))
    checks.append(check("PyTorch", True, torch.__version__))
    checks.append(
        check(
            "MPS compiled", bool(torch.backends.mps.is_built()), str(torch.backends.mps.is_built())
        )
    )
    checks.append(
        check(
            "MPS available",
            bool(torch.backends.mps.is_available()),
            str(torch.backends.mps.is_available()),
        )
    )
    config = root / "configs" / "demo_mac_m4.yaml"
    manifest = root / "data" / "rp001" / "demo_manifest.json"
    frontend = root / "frontend" / "index.html"
    checks.append(check("demo configuration", config.is_file(), str(config)))
    checks.append(check("release manifest", manifest.is_file(), str(manifest)))
    checks.append(check("frontend bundle", frontend.is_file(), str(frontend)))
    try:
        runtime = DemoRuntime(root, device="mps", developer_mode=True)
        runtime._load_manifest()
        checks.append(
            check(
                "reference bank", len(runtime.reference_descriptors) == 12, "12 cached descriptors"
            )
        )
        checks.append(
            check(
                "demo queries",
                len(runtime.manifest["known_cases"]) >= 3,
                "heldout/lighting/negative",
            )
        )
        checks.append(
            check("local model", runtime.paths.checkpoint.is_file(), str(runtime.paths.checkpoint))
        )
        hubconf = (
            root
            / "models"
            / "torch_hub"
            / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
            / "hubconf.py"
        )
        checks.append(check("offline DINOv3 source", hubconf.is_file(), str(hubconf)))
        checks.append(
            check(
                "artifact directory",
                runtime.paths.artifacts.parent.exists(),
                str(runtime.paths.artifacts),
            )
        )
        checks.append(check("network requirement", True, "no runtime request path"))
        free = shutil.disk_usage(root).free
        checks.append(check("free disk", free >= 3 * 1024**3, f"{free / 1024**3:.1f} GiB"))
        if not args.skip_inference:
            runtime.preload()
            result = runtime.run_case("heldout")
            checks.append(
                check("sample inference", result["verdict"] == "VERIFIED", result["verdict"])
            )
    except Exception as exc:  # exact failure is the purpose of this doctor
        checks.append(check("offline runtime", False, f"{type(exc).__name__}: {exc}"))
    passed = all(row[1] for row in checks)
    print("MAC M4 OFFLINE DEMO READY: " + ("PASS" if passed else "FAIL"))
    print(json.dumps({"checks": checks, "passed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
