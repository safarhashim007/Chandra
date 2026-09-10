"""Scientific-source checks and map-projected dense-GT quality for LROC products."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from rasterio.windows import Window
from shapely.geometry import Polygon

from chandrappan.data.gt_warp import dense_pixel_warp
from chandrappan.data.lroc_ingest import read_lroc_metadata
from chandrappan.data.manifest import ImageRecord
from chandrappan.geo.footprint import footprint_corners_world

SCIENTIFIC_RDR = "SCIENTIFIC_RDR"
BROWSE_ONLY = "BROWSE_ONLY"


def classify_source(path: str | Path, product_id: str) -> str:
    """Classify by actual file, never by a convenient fallback."""
    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"} or ".browse." in Path(path).name.lower():
        return BROWSE_ONLY
    if suffix in {".tif", ".tiff", ".img"} and product_id.upper().startswith(("NAC_PHO", "SDPPHO")):
        return SCIENTIFIC_RDR
    return BROWSE_ONLY


@dataclass(frozen=True)
class DenseGTQuality:
    valid_coverage: float
    median_cycle_error_px: float
    mean_cycle_error_px: float
    p95_cycle_error_px: float
    max_cycle_error_px: float
    valid_points: int
    accepted: bool
    rejection_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _valid_mask(path: Path, nodata: float | None, window: Window) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as raster:
        values = raster.read(1, window=window)
        valid = raster.read_masks(1, window=window) > 0
    valid &= np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    return valid


def _overlap_crop(metadata, center: tuple[float, float], size: int):
    x, y = metadata.transform.world_to_pixel(*center)
    origin_x = max(0, min(metadata.width - size, int(x - size / 2)))
    origin_y = max(0, min(metadata.height - size, int(y - size / 2)))
    world = metadata.transform.pixel_to_world(origin_x, origin_y)
    return (
        replace(
            metadata,
            width=size,
            height=size,
            transform=replace(
                metadata.transform, x_origin=float(world[0]), y_origin=float(world[1])
            ),
        ),
        Window(origin_x, origin_y, size, size),
    )


def dense_gt_quality(
    image_a: ImageRecord,
    image_b: ImageRecord,
    *,
    cycle_threshold_px: float = 0.25,
    min_valid_coverage: float = 0.05,
    crop_size: int = 640,
) -> DenseGTQuality:
    """Validate true A→B map GT, masks, and direct A→B→A cycle consistency."""
    if (
        classify_source(image_a.image_path, image_a.product_id) != SCIENTIFIC_RDR
        or classify_source(image_b.image_path, image_b.product_id) != SCIENTIFIC_RDR
    ):
        return DenseGTQuality(0, math.nan, math.nan, math.nan, math.nan, 0, False, "browse_only")
    source = read_lroc_metadata(image_a.image_path, Path(image_a.image_path).with_suffix(".xml"))
    target = read_lroc_metadata(image_b.image_path, Path(image_b.image_path).with_suffix(".xml"))
    if source.projection != target.projection:
        return DenseGTQuality(
            0, math.nan, math.nan, math.nan, math.nan, 0, False, "cross_crs_unsupported"
        )
    overlap = Polygon(footprint_corners_world(source)).intersection(
        Polygon(footprint_corners_world(target))
    )
    if overlap.is_empty:
        return DenseGTQuality(0, math.nan, math.nan, math.nan, math.nan, 0, False, "no_overlap")
    crop_size = min(crop_size, source.width, source.height, target.width, target.height)
    source, source_window = _overlap_crop(
        source, (overlap.centroid.x, overlap.centroid.y), crop_size
    )
    target, target_window = _overlap_crop(
        target, (overlap.centroid.x, overlap.centroid.y), crop_size
    )
    warp, inside = dense_pixel_warp(source, target)
    try:
        source_valid = _valid_mask(Path(image_a.image_path), image_a.nodata, source_window)
        target_valid = _valid_mask(Path(image_b.image_path), image_b.nodata, target_window)
    except Exception as exc:
        return DenseGTQuality(
            0,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            0,
            False,
            f"raster_read_error:{type(exc).__name__}",
        )
    target_x = np.floor(warp[..., 0]).astype(int)
    target_y = np.floor(warp[..., 1]).astype(int)
    target_ok = np.zeros_like(inside)
    target_ok[inside] = target_valid[target_y[inside], target_x[inside]]
    valid = inside & source_valid & target_ok
    if not valid.any():
        return DenseGTQuality(
            0, math.nan, math.nan, math.nan, math.nan, 0, False, "no_valid_dense_gt"
        )
    # Direct inverse world mapping evaluates B→A at each A→B subpixel point.
    world_x, world_y = target.transform.pixel_to_world(warp[..., 0], warp[..., 1])
    back_x, back_y = source.transform.world_to_pixel(world_x, world_y)
    grid_y, grid_x = np.indices((source.height, source.width), dtype=float)
    cycle = np.hypot(back_x - (grid_x + 0.5), back_y - (grid_y + 0.5))[valid]
    coverage = float(valid.mean())
    accepted = coverage >= min_valid_coverage and float(np.max(cycle)) <= cycle_threshold_px
    return DenseGTQuality(
        coverage,
        float(np.median(cycle)),
        float(np.mean(cycle)),
        float(np.percentile(cycle, 95)),
        float(np.max(cycle)),
        int(valid.sum()),
        accepted,
        None if accepted else "cycle_or_coverage_failure",
    )
