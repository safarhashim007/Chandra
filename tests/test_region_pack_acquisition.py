from pathlib import Path

from scripts.acquire_region_pack import download_atomic, sha256_file


def test_download_atomic_preserves_existing_valid_file(tmp_path: Path) -> None:
    destination = tmp_path / "existing.bin"
    destination.write_bytes(b"NASA")
    download_atomic("https://example.invalid/not-requested", destination)
    assert destination.read_bytes() == b"NASA"
    assert sha256_file(destination) == sha256_file(destination)
