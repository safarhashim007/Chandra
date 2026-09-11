"""Portable, inference-only runtime primitives.

This module owns device selection for the presentation path.  The production
RoMa inference implementation remains upstream; this adapter only makes its
runtime placement explicit and prevents a macOS launch from probing CUDA.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import torch

DeviceRequest = Literal["auto", "cuda", "mps", "cpu"]


@dataclass(frozen=True)
class RuntimeDevice:
    """Resolved inference device and its presentation-safe label."""

    torch_device: torch.device
    requested: DeviceRequest
    platform_name: str

    @property
    def ui_label(self) -> str:
        return {
            "cuda": "NVIDIA CUDA",
            "mps": "APPLE MPS",
            "cpu": "CPU FALLBACK",
        }[self.torch_device.type]


def resolve_device(requested: DeviceRequest = "auto") -> RuntimeDevice:
    """Resolve a device without ever attempting CUDA on macOS.

    ``auto`` keeps CUDA available for Ubuntu development while selecting MPS,
    then CPU, on a Mac.  Explicit unavailable devices are errors rather than
    silent device changes so the presentation operator sees the real state.
    """
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device request: {requested}")
    system = platform.system()
    mps_available = bool(torch.backends.mps.is_available())
    if requested == "auto":
        if system == "Darwin":
            selected = "mps" if mps_available else "cpu"
        elif torch.cuda.is_available():
            selected = "cuda"
        else:
            selected = "cpu"
    else:
        selected = requested
    if system == "Darwin" and selected == "cuda":
        raise RuntimeError("CUDA is not a supported Chandrappan Mac demo device")
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if selected == "mps" and not mps_available:
        raise RuntimeError("Apple MPS was requested but is unavailable")
    return RuntimeDevice(torch.device(selected), requested, system)


def _set_upstream_device(module: ModuleType, device: torch.device) -> None:
    """Patch upstream module globals imported by value before construction.

    RoMa v2 holds its selected device in several module globals.  No model
    weights or inference operators are changed here; this only makes a
    user-selected CPU fallback possible on a CUDA-capable host.
    """
    if hasattr(module, "device"):
        setattr(module, "device", device)


def load_official_roma(checkpoint: Path, runtime: RuntimeDevice):
    """Load the local official checkpoint in frozen inference mode.

    The caller must supply a local checkpoint.  This prevents RoMa's optional
    upstream download branch from being reachable in the presentation path.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"official RoMa v2 checkpoint is missing: {checkpoint}")
    # The checkpoint is always ``<release>/models/romav2_official.pt``.  Use
    # it as the relocation anchor rather than this module's import location:
    # source-tree tooling can therefore validate a copied release in place.
    release_root = checkpoint.resolve().parents[1]
    source_root = release_root / "vendor" / "romav2" / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"bundled RoMa v2 source is missing: {source_root}")
    hub_root = release_root / "models" / "torch_hub"
    hub_source = hub_root / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
    if not (hub_source / "hubconf.py").is_file():
        raise FileNotFoundError(
            "bundled DINOv3 Torch Hub source is missing; the offline demo will not download it"
        )
    torch.hub.set_dir(str(hub_root))
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import romav2

    for module_name in (
        "romav2.device",
        "romav2.geometry",
        "romav2.features",
        "romav2.matcher",
        "romav2.refiner",
        "romav2.dpt",
        "romav2.romav2",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            _set_upstream_device(module, runtime.torch_device)
    model = romav2.RoMaV2(checkpoint_path=str(checkpoint))
    model.apply_setting("base")
    model.to(runtime.torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if runtime.torch_device.type == "mps":
        # RoMa's upstream AMP path uses bfloat16.  MPS support varies across
        # PyTorch/macOS releases, so presentation inference uses portable
        # float32 rather than an approximation or a custom kernel.
        for module in model.modules():
            cfg = getattr(module, "cfg", None)
            if cfg is not None and hasattr(cfg, "enable_amp"):
                object.__setattr__(cfg, "enable_amp", False)
    return model


def release_device_memory(runtime: RuntimeDevice) -> None:
    """Release caches only after a completed job, never during matching."""
    if runtime.torch_device.type == "cuda":
        torch.cuda.empty_cache()
    elif runtime.torch_device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
