#!/usr/bin/env python3
"""CPU-safe project health checks; warnings do not masquerade as passes."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report(status: str, name: str, detail: str) -> bool:
    print(f"{status:<4} {name}: {detail}")
    return status != "FAIL"


def can_import(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def main() -> int:
    healthy = True
    healthy &= report(
        "PASS" if sys.version_info >= (3, 10) else "FAIL", "Python", sys.version.split()[0]
    )
    for module in ("numpy", "torch", "pytest", "sqlite3"):
        available = can_import(module)
        healthy &= report(
            "PASS" if available else "WARN", module, "available" if available else "not installed"
        )
    vendor = ROOT / "vendor" / "romav2" / "src" / "romav2"
    report(
        "PASS" if vendor.exists() else "WARN",
        "RoMa v2 source",
        str(vendor) if vendor.exists() else "not cloned",
    )
    checkpoint = Path(
        str(
            __import__("os").environ.get(
                "CHANDRAPPAN_ROMAV2_CHECKPOINT", ROOT / "checkpoints" / "romav2_official.pt"
            )
        )
    )
    report(
        "PASS" if checkpoint.exists() else "WARN",
        "RoMa v2 checkpoint",
        "available" if checkpoint.exists() else "not configured",
    )
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    healthy &= report("PASS" if free_gib >= 20 else "WARN", "Disk", f"{free_gib:.1f} GiB free")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
