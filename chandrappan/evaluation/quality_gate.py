"""Hard validation gate for permission to launch full fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QualityGateConfig:
    max_median_epe_increase_px: float = 0.0
    max_pck1_decrease: float = 0.0
    max_vrr_decrease: float = 0.0
    max_far_increase: float = 0.0
    min_pck3_or_pck5_increase: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_quality_gate(
    before: dict[str, float], after: dict[str, float], config: QualityGateConfig
) -> dict[str, object]:
    checks = {
        "median_epe_not_worse": after["median_epe_px"]
        <= before["median_epe_px"] + config.max_median_epe_increase_px,
        "pck1_not_materially_worse": after["pck_1"] >= before["pck_1"] - config.max_pck1_decrease,
        "vrr_not_materially_worse": after["vrr"] >= before["vrr"] - config.max_vrr_decrease,
        "far_not_materially_worse": after["far"] <= before["far"] + config.max_far_increase,
        "pck3_or_pck5_improved": max(
            after["pck_3"] - before["pck_3"], after["pck_5"] - before["pck_5"]
        )
        >= config.min_pck3_or_pck5_increase,
    }
    return {"passed": all(checks.values()), "checks": checks, "config": config.as_dict()}


def require_quality_gate(result: dict[str, object]) -> None:
    if not result["passed"]:
        raise RuntimeError("FULL TRAINING BLOCKED: validation quality gate failed.")
