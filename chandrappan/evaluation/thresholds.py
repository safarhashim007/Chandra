"""Validation-only registration threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .protocol import AcceptanceConfig, accept_registration


@dataclass(frozen=True)
class ThresholdSearchResult:
    config: AcceptanceConfig
    vrr: float
    far: float
    positive_count: int
    negative_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "vrr": self.vrr,
            "far": self.far,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }


def select_validation_thresholds(
    rows: list[dict[str, Any]],
    candidates: list[AcceptanceConfig],
    *,
    far_limit: float,
    dataset_name: str,
) -> ThresholdSearchResult:
    """Maximize validation VRR under FAR; reject T0 before inspecting rows."""
    if dataset_name.lower().startswith("t0"):
        raise ValueError("T0 is immutable and cannot be used for threshold selection")
    if not 0 <= far_limit <= 1:
        raise ValueError("far_limit must be between zero and one")
    positives = [row for row in rows if row["is_positive"]]
    negatives = [row for row in rows if not row["is_positive"]]
    if not positives or not negatives:
        raise ValueError("validation threshold selection requires positive and negative rows")
    results = []
    for config in candidates:
        accepted = [accept_registration(row["metrics"], config) for row in rows]
        positive_accepts = sum(
            accepted[index] for index, row in enumerate(rows) if row["is_positive"]
        )
        negative_accepts = sum(
            accepted[index] for index, row in enumerate(rows) if not row["is_positive"]
        )
        vrr = positive_accepts / len(positives)
        far = negative_accepts / len(negatives)
        if far <= far_limit:
            results.append(ThresholdSearchResult(config, vrr, far, len(positives), len(negatives)))
    if not results:
        raise ValueError("no validation threshold candidate satisfies FAR limit")
    return max(results, key=lambda result: (result.vrr, -result.far))
