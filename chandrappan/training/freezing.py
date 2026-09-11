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


def configure_stride1_refiner_training(model: nn.Module) -> nn.Module:
    """Freeze RoMa except its stride-1 refiner (the RP-001 D1 experiment).

    This is deliberately narrower than :func:`configure_refiner_only_training`.
    It exists so a small regional adaptation cannot silently optimize the
    backbone, matcher, or the stride-4/stride-2 refinement stages.
    """
    if not hasattr(model, "refiners"):
        raise TypeError("RoMa model must expose refiners")
    refiners = model.refiners
    try:
        stride1 = refiners["1"]
    except (KeyError, TypeError) as exc:
        raise TypeError("RoMa model must expose a stride-1 refiner named '1'") from exc
    freeze_module(model)
    unfreeze_module(stride1)
    return stride1
