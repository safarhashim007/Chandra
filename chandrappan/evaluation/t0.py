"""Immutable, checksummed project-defined T0 benchmark mechanics."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_checksum(path: str | Path, checksum_path: str | Path) -> str:
    checksum = sha256_file(path)
    destination = Path(checksum_path)
    if destination.exists() and destination.read_text(encoding="utf-8").strip() != checksum:
        raise RuntimeError(f"immutable benchmark checksum mismatch: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(checksum + "\n", encoding="utf-8")
    return checksum


def verify_checksum(path: str | Path, checksum_path: str | Path) -> None:
    expected = Path(checksum_path).read_text(encoding="utf-8").strip()
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"T0 checksum mismatch for {path}: {actual} != {expected}")


def reject_t0_for_training(
    dataset_path: str | Path, t0_path: str | Path, checksum_path: str | Path
) -> None:
    verify_checksum(t0_path, checksum_path)
    if Path(dataset_path).resolve() == Path(t0_path).resolve() or sha256_file(
        dataset_path
    ) == sha256_file(t0_path):
        raise RuntimeError("T0 benchmark is immutable and cannot be used for training")
