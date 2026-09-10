import torch

from chandrappan.training.coordinates import (
    gt_pixel_warp_to_roma,
    roma_warp_to_pixel,
)


def test_pixel_roma_pixel_roundtrip_corners_center_and_non_square() -> None:
    points = torch.tensor(
        [[0.0, 0.0], [639.0, 0.0], [0.0, 479.0], [639.0, 479.0], [320.0, 240.0], [37.25, 411.75]]
    )
    restored = roma_warp_to_pixel(gt_pixel_warp_to_roma(points, (480, 640)), (480, 640))
    assert torch.max(torch.abs(restored - points)).item() < 0.01
