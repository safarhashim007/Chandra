"""EMA restricted to the trainable RoMa refiner parameters."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


class RefinerEMA:
    """Maintain and temporarily evaluate an EMA without touching frozen modules."""

    def __init__(self, refiners: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = OrderedDict(
            (name, parameter.detach().clone()) for name, parameter in refiners.named_parameters()
        )

    def update(self, refiners: nn.Module) -> None:
        parameters = dict(refiners.named_parameters())
        if parameters.keys() != self.shadow.keys():
            raise ValueError("EMA refiner parameter set changed")
        with torch.no_grad():
            for name, parameter in parameters.items():
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @contextmanager
    def average_parameters(self, refiners: nn.Module) -> Iterator[None]:
        parameters = dict(refiners.named_parameters())
        if parameters.keys() != self.shadow.keys():
            raise ValueError("EMA refiner parameter set changed")
        saved = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(saved[name])
