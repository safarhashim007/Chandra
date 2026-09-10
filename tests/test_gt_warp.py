import numpy as np

from chandrappan.data.gt_warp import dense_pixel_warp
from chandrappan.geo.metadata import LunarImageMetadata
from chandrappan.geo.pixel_world import AffineTransform


def _metadata(origin_x: float, origin_y: float) -> LunarImageMetadata:
    return LunarImageMetadata(
        product_id=f"{origin_x}:{origin_y}",
        source_path="fixture.tif",
        width=2,
        height=2,
        center_lat=0,
        center_lon_east=0,
        gsd_m_per_px=1,
        transform=AffineTransform(origin_x, 1, 0, origin_y, 0, -1),
    )


def test_dense_warp_uses_pixel_centres_and_target_pixel_coordinates() -> None:
    warp, valid = dense_pixel_warp(_metadata(0, 10), _metadata(1, 10))
    expected = np.array([[[[-0.5, 0.5], [0.5, 0.5]], [[-0.5, 1.5], [0.5, 1.5]]]], dtype=np.float32)[
        0
    ]
    assert np.allclose(warp, expected)
    assert np.array_equal(valid, np.array([[False, True], [False, True]]))
