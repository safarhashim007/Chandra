import numpy as np
import pytest

from chandrappan.demo_runtime import spatially_distributed_matches
from chandrappan.runtime import resolve_device


def test_mac_auto_device_never_selects_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chandrappan.runtime.platform.system", lambda: "Darwin")
    assert resolve_device("auto").torch_device.type in {"mps", "cpu"}
    with pytest.raises(RuntimeError, match="not a supported"):
        resolve_device("cuda")


def test_display_matches_are_limited_and_spatially_distributed() -> None:
    source = np.array([[20, 20], [600, 20], [20, 600], [600, 600], [300, 300]], dtype=np.float32)
    target = source.copy()
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.99], dtype=np.float32)
    verification = {"selected": {"matrix": [[1, 0, 0], [0, 1, 0]]}}
    matches = spatially_distributed_matches(source, target, confidence, verification, 640, 4)
    assert len(matches) == 4
    assert len({tuple(match["query"]) for match in matches}) == 4
    assert matches[0]["id"] == "M01"
