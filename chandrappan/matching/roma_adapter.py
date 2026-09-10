"""RoMa v2 adapter that preserves official inference rather than reimplementing it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MatchResult:
    warp_ab: Any
    warp_ba: Any
    overlap_ab: Any
    overlap_ba: Any
    precision_ab: Any
    precision_ba: Any
    raw: dict[str, Any]


class RoMaAdapter:
    def __init__(self, model: Any):
        self.model = model

    def match(self, image_a: Any, image_b: Any) -> MatchResult:
        predictions = self.model.match(image_a, image_b)
        required = {
            "warp_AB",
            "warp_BA",
            "overlap_AB",
            "overlap_BA",
            "precision_AB",
            "precision_BA",
        }
        missing = required.difference(predictions)
        if missing:
            raise RuntimeError(f"RoMa v2 output contract changed; missing {sorted(missing)}")
        return MatchResult(
            warp_ab=predictions["warp_AB"],
            warp_ba=predictions["warp_BA"],
            overlap_ab=predictions["overlap_AB"],
            overlap_ba=predictions["overlap_BA"],
            precision_ab=predictions["precision_AB"],
            precision_ba=predictions["precision_BA"],
            raw=predictions,
        )
