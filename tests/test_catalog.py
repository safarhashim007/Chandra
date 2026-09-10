import pytest

pytest.importorskip("sqlite3", reason="host Python lacks the _sqlite3 extension")

from chandrappan.geo.catalog import LunarCatalog
from chandrappan.geo.metadata import LunarImageMetadata
from chandrappan.geo.pixel_world import AffineTransform


def _metadata(product_id: str, origin_x: float) -> LunarImageMetadata:
    return LunarImageMetadata(
        product_id=product_id,
        source_path=f"{product_id}.tif",
        width=100,
        height=100,
        center_lat=0,
        center_lon_east=0,
        gsd_m_per_px=1,
        transform=AffineTransform(origin_x, 1, 0, 100, 0, -1),
    )


def test_catalog_rtree_overlapping_query(tmp_path) -> None:
    catalog = LunarCatalog(tmp_path / "catalog.sqlite")
    catalog.add(_metadata("a", 0))
    catalog.add(_metadata("b", 1000))
    assert [row["product_id"] for row in catalog.query_bounds(50, 75, 10, 50)] == ["a"]
    catalog.close()
