#!/usr/bin/env python3
"""Report the actual execution environment without assuming CUDA availability."""

from __future__ import annotations

import platform
import shutil
import sys


def main() -> int:
    print(f"OS: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    total, used, free = shutil.disk_usage(".")
    print(f"Disk free: {free / 2**30:.1f} GiB / {total / 2**30:.1f} GiB")
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
        print(f"MPS available: {mps_available}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                print(f"GPU {index}: {properties.name}; {properties.total_memory / 2**20:.0f} MiB")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
