from chandrappan.geo.footprint import footprint_corners_world
from chandrappan.geo.metadata import LunarImageMetadata
from chandrappan.geo.overlap import convex_intersection
from chandrappan.geo.pixel_world import AffineTransform


def _metadata(origin_x: float) -> LunarImageMetadata:
    return LunarImageMetadata(
        product_id=str(origin_x),
        source_path="fixture.tif",
        width=100,
        height=100,
        center_lat=0,
        center_lon_east=0,
        gsd_m_per_px=1,
        transform=AffineTransform(origin_x, 1, 0, 100, 0, -1),
    )


def test_north_up_footprints_are_counter_clockwise_for_intersection() -> None:
    first = footprint_corners_world(_metadata(0))
    second = footprint_corners_world(_metadata(50))
    assert convex_intersection(first, second)
