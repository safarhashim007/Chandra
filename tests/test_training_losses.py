import torch

from chandrappan.training.losses import roma_robust_warp_loss, total_refiner_loss, warp_huber_loss
from chandrappan.training.objective import refiner_stage_losses


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


def test_roma_robust_loss_is_source_form_and_mask_safe() -> None:
    target = torch.zeros(1, 1, 1, 2)
    predicted = torch.tensor([[[[0.004, 0.0]]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    loss = roma_robust_warp_loss(predicted, target, valid, stride=4)
    cs = 1e-3 * 4
    expected = cs**0.5 * ((0.004 / cs) ** 2 + 1.0) ** 0.25
    assert torch.isclose(loss, torch.tensor(expected))
    assert roma_robust_warp_loss(predicted, target, ~valid, stride=4).item() == 0.0


def test_huber_beta_has_pixel_scale_at_640() -> None:
    target = torch.zeros(1, 1, 1, 2)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    # A normalized x residual of 1.0 is 320 pixels at width 640. Smooth-L1
    # beta therefore acts in normalized, not pixel, units.
    beta_one = warp_huber_loss(target + torch.tensor([[[[1.0, 0.0]]]]), target, valid, beta=1.0)
    beta_small = warp_huber_loss(target + torch.tensor([[[[0.01, 0.0]]]]), target, valid, beta=0.01)
    assert torch.isclose(beta_one, torch.tensor(0.25))
    assert torch.isclose(beta_small, torch.tensor(0.0025))


def test_stagewise_objective_gives_each_detached_refiner_a_gradient() -> None:
    master = torch.zeros(1, 8, 12, 2)
    valid = torch.ones(1, 8, 12, dtype=torch.bool)
    stage4 = torch.nn.Parameter(torch.full((1, 2, 3, 2), 0.1))
    stage2 = torch.nn.Parameter(torch.full((1, 4, 6, 2), 0.1))
    stage1 = torch.nn.Parameter(torch.full((1, 8, 12, 2), 0.1))
    result = refiner_stage_losses(
        [
            {"stride": 4, "ab": {"warp": stage4}},
            {"stride": 2, "ab": {"warp": stage2}},
            {"stride": 1, "ab": {"warp": stage1}},
        ],
        master,
        valid,
        (8, 12),
        (8, 12),
    )
    result["total"].backward()
    assert all(
        stage.grad is not None and torch.count_nonzero(stage.grad)
        for stage in (stage4, stage2, stage1)
    )


def test_robust_gradient_step_reduces_pixel_epe_for_axis_and_diagonal_offsets() -> None:
    for offset in ((0.5, 0.0), (0.0, -1.0), (5.0, -5.0)):
        target = torch.zeros(1, 1, 1, 2)
        prediction = torch.nn.Parameter(
            torch.tensor(offset).view(1, 1, 1, 2) * 2 / torch.tensor([640.0, 480.0])
        )
        before = torch.linalg.vector_norm(
            (prediction - target) * torch.tensor([320.0, 240.0])
        ).item()
        loss = roma_robust_warp_loss(
            prediction, target, torch.ones(1, 1, 1, dtype=torch.bool), stride=1
        )
        loss.backward()
        with torch.no_grad():
            prediction.add_(prediction.grad, alpha=-1e-5)
        after = torch.linalg.vector_norm(
            (prediction - target) * torch.tensor([320.0, 240.0])
        ).item()
        assert after < before
