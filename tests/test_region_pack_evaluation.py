import torch

from scripts.evaluate_region_pack import _points


def test_bidirectional_point_filter_keeps_identity_correspondences() -> None:
    height = width = 4
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    warp = torch.stack((2 * x / width - 1, 2 * y / height - 1), dim=-1)[None]
    prediction = {
        "warp_AB": warp,
        "warp_BA": warp.clone(),
        "overlap_AB": torch.ones((1, height, width, 1)),
        "precision_AB": torch.eye(2).expand(1, height, width, 2, 2).clone(),
    }
    source, target, _, diagnostics = _points(prediction, crop_size=40, max_matches=32)
    assert len(source) == height * width
    assert torch.tensor(source).allclose(torch.tensor(target))
    assert diagnostics["cycle_median_px"] == 0.0
