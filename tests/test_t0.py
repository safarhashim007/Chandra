from pathlib import Path

import pytest

from chandrappan.evaluation.t0 import freeze_checksum, reject_t0_for_training, verify_checksum
from chandrappan.training.preflight import assert_training_allowed


def test_t0_checksum_is_immutable_and_rejected_for_training(tmp_path: Path) -> None:
    t0 = tmp_path / "T0.parquet"
    checksum = tmp_path / "T0.sha256"
    t0.write_bytes(b"frozen")
    freeze_checksum(t0, checksum)
    verify_checksum(t0, checksum)
    with pytest.raises(RuntimeError, match="cannot be used for training"):
        reject_t0_for_training(t0, t0, checksum)
    t0.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_checksum(t0, checksum)


def test_training_preflight_rejects_failed_quality_gate(tmp_path: Path) -> None:
    t0 = tmp_path / "T0.parquet"
    checksum = tmp_path / "T0.sha256"
    dataset = tmp_path / "train.parquet"
    t0.write_bytes(b"frozen")
    dataset.write_bytes(b"train")
    freeze_checksum(t0, checksum)
    with pytest.raises(RuntimeError, match="FULL TRAINING BLOCKED"):
        assert_training_allowed(dataset, t0, checksum, {"passed": False})
