import pytest
import torch
from torch import nn

from chandrappan.training.ema import RefinerEMA
from chandrappan.training.freezing import configure_refiner_only_training
from chandrappan.training.preflight import reject_t0_pair


def test_ema_tracks_only_refiners_and_restores_raw_parameters() -> None:
    refiners = nn.Linear(2, 2, bias=False)
    ema = RefinerEMA(refiners, decay=0.999)
    before = refiners.weight.detach().clone()
    with torch.no_grad():
        refiners.weight.add_(1.0)
    raw = refiners.weight.detach().clone()
    ema.update(refiners)
    with ema.average_parameters(refiners):
        assert torch.allclose(refiners.weight, before + 0.001)
    assert torch.equal(refiners.weight, raw)


def test_refiner_only_setup_leaves_backbone_frozen() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.refiners = nn.Linear(2, 2)

    model = Model()
    configure_refiner_only_training(model)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.refiners.parameters())


def test_t0_pair_is_rejected_from_training_diagnostics() -> None:
    with pytest.raises(RuntimeError, match="immutable"):
        reject_t0_pair(
            "NAC_PHO_E018N3346_M107042466L__NAC_PHO_E018N3346_M107042466R",
            "benchmarks/T0_v1.parquet",
        )
