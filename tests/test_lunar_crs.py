from chandrappan.geo.lunar_crs import (
    east_to_west_longitude,
    normalize_east_longitude,
    validate_latitude,
    west_to_east_longitude,
)


def test_east_west_longitude_roundtrip() -> None:
    for value in (0.0, 10.0, 180.0, 359.5):
        assert (
            normalize_east_longitude(west_to_east_longitude(east_to_west_longitude(value))) == value
        )


def test_latitude_validation() -> None:
    assert validate_latitude(-90) == -90
    assert validate_latitude(90) == 90
