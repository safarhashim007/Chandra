#!/usr/bin/env python3
"""Inspect source/API safely; --run additionally downloads/loads the official checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_SOURCE = ROOT / "vendor" / "romav2" / "src"


def source_report() -> dict[str, object]:
    return {
        "source_root": str(VENDOR_SOURCE),
        "source_present": VENDOR_SOURCE.exists(),
        "model_class": "romav2.romav2.RoMaV2",
        "inference_entry": "RoMaV2.forward (torch.inference_mode)",
        "matcher": "romav2.matcher.Matcher",
        "descriptor": "romav2.features.Descriptor",
        "fine_features": "romav2.features.FineFeatures",
        "refiners": [4, 2, 1],
        "warp_convention": "normalized target x/y coordinates in [-1, 1]; align_corners=False",
        "confidence": "channel 0 overlap logit; channels 1:4 precision parameters",
        "checkpoint_load": "torch.hub.load_state_dict_from_url in RoMaV2.__init__",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="store_true", help="load checkpoint and run a random smoke forward"
    )
    parser.add_argument("--checkpoint", type=Path, help="optional local official checkpoint")
    parser.add_argument("--setting", default="turbo", choices=("turbo", "fast", "base", "precise"))
    args = parser.parse_args()
    report = source_report()
    if args.run:
        sys.path.insert(0, str(VENDOR_SOURCE))
        import torch
        from romav2 import RoMaV2

        model = RoMaV2(checkpoint_path=str(args.checkpoint) if args.checkpoint else None)
        model.apply_setting(args.setting)
        sample = torch.rand(1, 3, model.H_lr, model.W_lr, device=next(model.parameters()).device)
        predictions = model(sample, sample)
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        report["trainable_parameter_count"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        report["output_shapes"] = {
            key: list(value.shape) for key, value in predictions.items() if hasattr(value, "shape")
        }
        report["output_ranges"] = {
            key: {
                "min": float(value.min().item()),
                "max": float(value.max().item()),
                "finite": bool(torch.isfinite(value).all().item()),
            }
            for key, value in predictions.items()
            if hasattr(value, "shape") and value is not None
        }
    output = ROOT / "artifacts" / "romav2_inspection.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
