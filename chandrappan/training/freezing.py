"""Centralized trainability controls used by all future LunarRoMa stages."""

from __future__ import annotations

import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def assert_frozen(module: nn.Module) -> None:
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise AssertionError("module expected frozen but has trainable parameters")


def assert_trainable(module: nn.Module) -> None:
    if not any(parameter.requires_grad for parameter in module.parameters()):
        raise AssertionError("module expected trainable but has no trainable parameters")


def configure_refiner_only_training(model: nn.Module) -> None:
    """Freeze RoMa globally, then expose only its refiner heads to optimization."""
    if not hasattr(model, "refiners"):
        raise TypeError("RoMa model must expose refiners")
    freeze_module(model)
    unfreeze_module(model.refiners)
