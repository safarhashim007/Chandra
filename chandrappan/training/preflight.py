"""Mandatory checks before any full LunarRoMa training launch."""

from __future__ import annotations

from pathlib import Path

from chandrappan.evaluation.quality_gate import require_quality_gate
from chandrappan.evaluation.t0 import reject_t0_for_training


def assert_training_allowed(
    dataset_path: str | Path,
    t0_path: str | Path,
    t0_checksum_path: str | Path,
    quality_gate_result: dict[str, object],
) -> None:
    """Reject T0 leakage or failed validation quality before full training."""
    reject_t0_for_training(dataset_path, t0_path, t0_checksum_path)
    require_quality_gate(quality_gate_result)


def reject_t0_pair(pair_id: str, t0_pairs_path: str | Path) -> None:
    """Reject a pair identity contained in the immutable held-out benchmark."""
    from chandrappan.data.pairs import read_pairs

    if pair_id in {pair.pair_id for pair in read_pairs(t0_pairs_path)}:
        raise RuntimeError("T0 data is immutable and cannot be used for training diagnostics")
