import numpy as np

from chandrappan.geo.pixel_world import AffineTransform


def test_pixel_world_roundtrip_non_square_rotated() -> None:
    transform = AffineTransform(100.0, 2.0, -0.2, -50.0, 0.3, -1.7)
    x = np.array([0.0, 639.0, 17.4, 320.5])
    y = np.array([0.0, 479.0, 301.2, 240.0])
    world_x, world_y = transform.pixel_to_world(x, y)
    recovered_x, recovered_y = transform.world_to_pixel(world_x, world_y)
    assert np.max(np.abs(recovered_x - x)) < 1e-10
    assert np.max(np.abs(recovered_y - y)) < 1e-10
