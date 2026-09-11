import json

from chandrappan.data.lroc_ode import (
    discover_sites,
    parse_ode_products,
    select_acquisition_products,
)


def _product(title: str, tif_kbytes: int = 10) -> dict:
    return {
        "Product_title": title,
        "Product_lid": f"urn:test:{title.lower()}",
        "Product_creation_time": "2026-01-01T00:00:00",
        "Product_files": {
            "Product_file": [
                {
                    "FileName": f"{title}.TIF",
                    "KBytes": str(tif_kbytes),
                    "URL": f"https://example.test/{title}.TIF",
                    "Description": "FULL BROWSE IMAGE",
                },
                {
                    "FileName": f"{title}.XML",
                    "KBytes": "1",
                    "URL": f"https://example.test/{title}.xml",
                    "Description": "PDS4 PRODUCT LABEL FILE",
                },
                {
                    "FileName": f"{title}.IMG",
                    "KBytes": "100",
                    "URL": f"https://example.test/{title}.IMG",
                    "Description": "PRODUCT DATA FILE",
                },
                {
                    "FileName": f"{title}.MASK.TIF",
                    "KBytes": "1",
                    "URL": f"https://example.test/{title}.MASK.TIF",
                    "Description": "MASK IMAGE",
                },
            ]
        },
    }


def test_parse_ode_products_groups_sdppho_regions() -> None:
    payload = {
        "ODEResults": {
            "Products": {
                "Product": [
                    _product("NAC_PHO_E207N3357_M102314667L", 37),
                    _product("NAC_PHO_E207N3357_M109392442R", 38),
                    _product("NAC_PHO_E090S0155_M102057602R", 30),
                ]
            }
        }
    }

    products = parse_ode_products(json.dumps(payload))
    sites = discover_sites(products)

    assert [product.region_id for product in products] == [
        "E207N3357",
        "E207N3357",
        "E090S0155",
    ]
    site = next(row for row in sites if row.site_id == "E207N3357")
    assert site.product_count == 2
    assert site.candidate_positive_pairs == 1
    assert site.has_masks


def test_select_acquisition_products_prefers_small_geotiffs_per_region() -> None:
    products = parse_ode_products(
        {
            "ODEResults": {
                "Products": {
                    "Product": [
                        _product("NAC_PHO_E199N0308_M1L", 90),
                        _product("NAC_PHO_E199N0308_M2L", 10),
                        _product("NAC_PHO_E199N0308_M3L", 20),
                    ]
                }
            }
        }
    )

    selected = select_acquisition_products(products, regions=["E199N0308"], products_per_region=2)

    assert [product.product_id for product in selected] == [
        "NAC_PHO_E199N0308_M2L",
        "NAC_PHO_E199N0308_M3L",
    ]
