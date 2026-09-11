"""Official ODE/PDS helpers for LROC SDPPHO discovery and acquisition."""

from __future__ import annotations

import json
import re
import shutil
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ODE_SDPPHO_QUERY = (
    "https://oderest.rsl.wustl.edu/live2/"
    "?query=product&target=moon&ihid=LRO&iid=LROC&pt=SDPPHO&results=fp&output=json"
)


@dataclass(frozen=True)
class OdeProduct:
    product_id: str
    region_id: str
    label_url: str
    product_img_url: str | None
    geotiff_url: str
    mask_url: str | None
    browse_png_url: str | None
    geotiff_kbytes: int
    product_img_kbytes: int | None
    product_lid: str | None
    product_creation_time: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SiteDiscovery:
    site_id: str
    product_count: int
    product_ids: list[str]
    geotiff_kbytes_total: int
    product_img_kbytes_total: int
    candidate_positive_pairs: int
    has_masks: bool
    acquisition_priority: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _region_from_title(title: str) -> str:
    match = re.match(r"NAC_PHO_(E\d+[NS]\d+)_", title.upper())
    if not match:
        raise ValueError(f"unsupported SDPPHO product title: {title}")
    return match.group(1)


def _file_by_name(files: list[dict[str, Any]], filename: str) -> dict[str, Any] | None:
    wanted = filename.upper()
    return next((row for row in files if str(row.get("FileName", "")).upper() == wanted), None)


def _file_by_description(files: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
    wanted = text.upper()
    return next((row for row in files if wanted in str(row.get("Description", "")).upper()), None)


def parse_ode_products(payload: str | bytes | dict[str, Any]) -> list[OdeProduct]:
    """Parse ODE SDPPHO product JSON into stable acquisition records."""
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    products = _as_list(data["ODEResults"]["Products"]["Product"])
    parsed: list[OdeProduct] = []
    for product in products:
        title = product["Product_title"].upper()
        files = _as_list(product.get("Product_files", {}).get("Product_file"))
        geotiff = _file_by_name(files, f"{title}.TIF")
        label = _file_by_name(files, f"{title}.XML")
        product_img = _file_by_name(files, f"{title}.IMG")
        mask = _file_by_name(files, f"{title}.MASK.TIF")
        browse_png = _file_by_name(files, f"{title}.BROWSE.PNG")
        if geotiff is None:
            continue
        if label is None:
            label = _file_by_description(files, "PRODUCT LABEL")
        if label is None:
            raise ValueError(f"missing ODE label URL for {title}")
        parsed.append(
            OdeProduct(
                product_id=title,
                region_id=_region_from_title(title),
                label_url=str(label["URL"]),
                product_img_url=str(product_img["URL"]) if product_img else None,
                geotiff_url=str(geotiff["URL"]),
                mask_url=str(mask["URL"]) if mask else None,
                browse_png_url=str(browse_png["URL"]) if browse_png else None,
                geotiff_kbytes=int(geotiff.get("KBytes") or 0),
                product_img_kbytes=int(product_img["KBytes"]) if product_img else None,
                product_lid=product.get("Product_lid"),
                product_creation_time=product.get("Product_creation_time"),
            )
        )
    return parsed


def discover_sites(products: list[OdeProduct]) -> list[SiteDiscovery]:
    """Group official products into repeat-observation site summaries."""
    by_region: dict[str, list[OdeProduct]] = {}
    for product in products:
        by_region.setdefault(product.region_id, []).append(product)
    sites = []
    for region_id, rows in by_region.items():
        count = len(rows)
        candidate_pairs = count * (count - 1) // 2
        geotiff_total = sum(row.geotiff_kbytes for row in rows)
        img_total = sum(row.product_img_kbytes or 0 for row in rows)
        # More repeat observations are best; smaller GeoTIFFs are cheaper to validate early.
        priority = candidate_pairs / max(1.0, geotiff_total / 1024)
        sites.append(
            SiteDiscovery(
                site_id=region_id,
                product_count=count,
                product_ids=sorted(row.product_id for row in rows),
                geotiff_kbytes_total=geotiff_total,
                product_img_kbytes_total=img_total,
                candidate_positive_pairs=candidate_pairs,
                has_masks=all(row.mask_url is not None for row in rows),
                acquisition_priority=priority,
            )
        )
    return sorted(sites, key=lambda row: (-row.acquisition_priority, row.site_id))


def select_acquisition_products(
    products: list[OdeProduct],
    *,
    regions: list[str],
    products_per_region: int,
) -> list[OdeProduct]:
    """Choose a small incremental product set from repeat sites."""
    by_region: dict[str, list[OdeProduct]] = {}
    for product in products:
        by_region.setdefault(product.region_id, []).append(product)
    selected = []
    for region in regions:
        candidates = sorted(
            by_region.get(region, []),
            key=lambda row: (row.geotiff_kbytes, row.product_id),
        )
        selected.extend(candidates[:products_per_region])
    return selected


def download_product_files(
    products: list[OdeProduct],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Download GeoTIFF/XML pairs and return provenance rows."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    provenance = []
    for product in products:
        rows = [
            (product.geotiff_url, target / f"{product.product_id}.TIF", "geotiff"),
            (product.label_url, target / f"{product.product_id}.xml", "label"),
        ]
        for url, path, kind in rows:
            if overwrite or not path.exists() or path.stat().st_size == 0:
                with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as out:
                    shutil.copyfileobj(response, out)
            provenance.append(
                {
                    "product_id": product.product_id,
                    "region_id": product.region_id,
                    "kind": kind,
                    "path": str(path),
                    "url": url,
                    "bytes": path.stat().st_size,
                    "ode_product_lid": product.product_lid,
                    "full_product_img_url": product.product_img_url,
                    "mask_url": product.mask_url,
                    "source_note": (
                        "Official ODE/PDS SDPPHO product record; local raster is the "
                        "PDS full browse GeoTIFF representation, with the full IMG "
                        "product URL preserved in provenance."
                    ),
                }
            )
    return provenance


def site_counts(products: list[OdeProduct]) -> Counter[str]:
    return Counter(product.region_id for product in products)
