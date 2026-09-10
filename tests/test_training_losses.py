import torch

from chandrappan.training.losses import total_refiner_loss, warp_huber_loss


def test_warp_loss_is_lower_for_exact_prediction() -> None:
    target = torch.zeros(1, 3, 3, 2)
    valid = torch.ones(1, 3, 3, dtype=torch.bool)
    exact = warp_huber_loss(target, target, valid)
    shifted = warp_huber_loss(target + 0.1, target, valid)
    assert exact.item() == 0.0
    assert shifted > exact


def test_negative_pair_disables_warp_but_supervises_overlap() -> None:
    target = torch.zeros(1, 2, 2, 2)
    result = total_refiner_loss(
        target + 100,
        target,
        torch.zeros(1, 2, 2, dtype=torch.bool),
        torch.full((1, 2, 2), 10.0),
        torch.zeros(1, 2, 2),
    )
    assert result["warp"].item() == 0.0
    assert result["overlap"] > 1
