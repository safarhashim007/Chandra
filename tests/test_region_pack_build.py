import pytest

from scripts.build_region_pack import validate_acquisition_split


def test_region_pack_split_is_acquisition_isolated() -> None:
    products = [
        {"product_id": f"P{index}", "role": role}
        for index, role in enumerate(["TRAIN"] * 12 + ["VALIDATION"] * 4 + ["TEST"] * 4)
    ]
    split = validate_acquisition_split(products)
    assert {key: len(value) for key, value in split.items()} == {
        "TRAIN": 12,
        "VALIDATION": 4,
        "TEST": 4,
    }


def test_region_pack_split_rejects_duplicate_acquisition() -> None:
    products = [
        {"product_id": "P0", "role": "TRAIN"},
        {"product_id": "P0", "role": "TEST"},
    ] + [{"product_id": f"P{index}", "role": "TRAIN"} for index in range(1, 15)]
    with pytest.raises(ValueError, match="leakage"):
        validate_acquisition_split(products)
