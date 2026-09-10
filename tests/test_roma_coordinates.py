import torch

from chandrappan.training.coordinates import (
    build_gt_for_stage,
    gt_pixel_warp_to_roma,
    normalized_delta_to_pixel,
    refiner_head_delta_to_pixel,
    roma_warp_to_pixel,
)


def test_pixel_roma_pixel_roundtrip_corners_center_and_non_square() -> None:
    points = torch.tensor(
        [[0.0, 0.0], [639.0, 0.0], [0.0, 479.0], [639.0, 479.0], [320.0, 240.0], [37.25, 411.75]]
    )
    restored = roma_warp_to_pixel(gt_pixel_warp_to_roma(points, (480, 640)), (480, 640))
    assert torch.max(torch.abs(restored - points)).item() < 0.01


def test_roma_uses_pixel_centres_like_official_grid() -> None:
    pixel_center = torch.tensor([[0.5, 0.5]])
    roma = gt_pixel_warp_to_roma(pixel_center, (480, 640))
    expected = torch.tensor([[-1.0 + 1.0 / 640, -1.0 + 1.0 / 480]])
    assert torch.allclose(roma, expected)


def test_pixel_roma_pixel_roundtrip_is_subpixel_for_square_and_non_square_shapes() -> None:
    for size in ((640, 640), (480, 640), (319, 701)):
        height, width = size
        points = torch.tensor(
            [[0.5, 0.5], [width - 0.5, height - 0.5], [width / 2.0, height / 2.0], [17.125, 23.875]]
        )
        restored = roma_warp_to_pixel(gt_pixel_warp_to_roma(points, size), size)
        assert torch.max(torch.abs(restored - points)).item() < 1e-5


def test_x_and_y_normalized_delta_scales_are_independent() -> None:
    pixels = normalized_delta_to_pixel(torch.tensor([[0.1, 0.1]]), (480, 640))
    assert torch.allclose(pixels, torch.tensor([[32.0, 24.0]]))


def test_refiner_head_delta_is_one_pixel_at_every_roma_stride() -> None:
    for stride in (4, 2, 1):
        assert stride in {1, 2, 4}  # Stride changes the grid, not head units.
        pixels = refiner_head_delta_to_pixel(torch.tensor([[8.0, -8.0]]))
        assert torch.allclose(pixels, torch.tensor([[1.0, -1.0]]))


def test_stage_gt_sampling_preserves_an_affine_non_square_warp() -> None:
    height, width = 480, 640
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    warp = torch.stack((2.0 * x + 0.5 * y + 3.0, -0.25 * x + 1.5 * y + 7.0), dim=-1).float()[None]
    target, valid = build_gt_for_stage(
        warp,
        torch.ones(1, height, width, dtype=torch.bool),
        (height, width),
        (height, width),
        (120, 160),
    )
    restored = roma_warp_to_pixel(target, (height, width))
    source_x = (torch.arange(160) + 0.5) * width / 160 - 0.5
    source_y = (torch.arange(120) + 0.5) * height / 120 - 0.5
    yy, xx = torch.meshgrid(source_y, source_x, indexing="ij")
    expected = torch.stack((2.0 * xx + 0.5 * yy + 3.0, -0.25 * xx + 1.5 * yy + 7.0), dim=-1)
    assert valid.all()
    assert torch.max(torch.abs(restored[0] - expected)).item() < 3e-4
