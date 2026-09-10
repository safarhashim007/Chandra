import torch.nn as nn

from chandrappan.training.freezing import (
    assert_frozen,
    assert_trainable,
    count_trainable_parameters,
    freeze_module,
    unfreeze_module,
)


def test_freezing_helpers_control_and_count_parameters() -> None:
    module = nn.Linear(3, 2)
    assert count_trainable_parameters(module) == 8
    freeze_module(module)
    assert_frozen(module)
    assert count_trainable_parameters(module) == 0
    unfreeze_module(module)
    assert_trainable(module)
    assert count_trainable_parameters(module) == 8
