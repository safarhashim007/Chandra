"""Deterministic Region Pack primitives used by the production orchestration.

Regional retrieval is only a candidate proposal stage.  Nothing in this module
returns a registration verdict; that decision remains with bidirectional RoMa
matching, geometric verification, and the configured acceptance policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalCandidate:
    product_id: str
    score: float
    rank: int

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class NccRefinement:
    accepted: bool
    dx_px: float | None
    dy_px: float | None
    peak: float | None
    peak_margin: float | None
    rejection_reason: str | None

    def as_dict(self) -> dict[str, float | bool | str | None]:
        return asdict(self)


def intensity_descriptor(values: np.ndarray, *, bins: int = 64) -> np.ndarray:
    """Return a finite, L2-normalised I/F histogram without learned dependencies."""
    image = np.asarray(values, dtype=np.float64)
    finite = image[np.isfinite(image)]
    if finite.size < 16:
        raise ValueError("descriptor requires at least 16 finite pixels")
    low, high = np.quantile(finite, (0.01, 0.99))
    if not high > low:
        raise ValueError("descriptor image has no usable intensity range")
    histogram, _ = np.histogram(np.clip(finite, low, high), bins=bins, range=(low, high))
    descriptor = histogram.astype(np.float64)
    norm = np.linalg.norm(descriptor)
    if norm == 0:
        raise ValueError("descriptor unexpectedly has zero norm")
    return (descriptor / norm).astype(np.float32)


def rank_references(
    query_descriptor: np.ndarray, reference_descriptors: dict[str, np.ndarray]
) -> list[RetrievalCandidate]:
    """Rank the complete eligible regional bank by cosine similarity, stably."""
    query = np.asarray(query_descriptor, dtype=np.float64).reshape(-1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0 or not np.isfinite(query).all():
        raise ValueError("query descriptor must be finite and non-zero")
    scored = []
    for product_id, descriptor in reference_descriptors.items():
        reference = np.asarray(descriptor, dtype=np.float64).reshape(-1)
        if reference.shape != query.shape:
            raise ValueError(f"descriptor shape mismatch for {product_id}")
        norm = np.linalg.norm(reference)
        if norm == 0 or not np.isfinite(reference).all():
            raise ValueError(f"reference descriptor is invalid for {product_id}")
        scored.append((float(np.dot(query, reference) / (query_norm * norm)), product_id))
    return [
        RetrievalCandidate(product_id=product_id, score=score, rank=index)
        for index, (score, product_id) in enumerate(
            sorted(scored, key=lambda row: (-row[0], row[1])), start=1
        )
    ]


def fractional_zncc_peak(
    correlation: np.ndarray,
    *,
    min_peak: float = 0.2,
    min_margin: float = 0.03,
) -> NccRefinement:
    """Estimate a fractional ZNCC peak and reject flat, clipped, or ambiguous peaks.

    ``dx_px`` and ``dy_px`` are relative to the integer correlation-map centre,
    not absolute image coordinates.  The caller must still perform the reverse
    refinement consistency test before using this estimate.
    """
    values = np.asarray(correlation, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 3:
        raise ValueError("correlation must be a two-dimensional map of at least 3 by 3")
    if not np.isfinite(values).all():
        return NccRefinement(False, None, None, None, None, "non_finite_correlation")
    y, x = np.unravel_index(np.argmax(values), values.shape)
    peak = float(values[y, x])
    if x in (0, values.shape[1] - 1) or y in (0, values.shape[0] - 1):
        return NccRefinement(False, None, None, peak, None, "border_clipped_peak")
    competitors = values.copy()
    competitors[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = -np.inf
    margin = peak - float(np.max(competitors))
    if peak < min_peak:
        return NccRefinement(False, None, None, peak, margin, "low_zncc_peak")
    if margin < min_margin:
        return NccRefinement(False, None, None, peak, margin, "ambiguous_zncc_peak")
    x_left, x_center, x_right = values[y, x - 1], values[y, x], values[y, x + 1]
    y_top, y_center, y_bottom = values[y - 1, x], values[y, x], values[y + 1, x]
    x_denom = x_left - 2 * x_center + x_right
    y_denom = y_top - 2 * y_center + y_bottom
    if abs(x_denom) < 1e-12 or abs(y_denom) < 1e-12:
        return NccRefinement(False, None, None, peak, margin, "flat_zncc_peak")
    dx = float((x - (values.shape[1] - 1) / 2) + 0.5 * (x_left - x_right) / x_denom)
    dy = float((y - (values.shape[0] - 1) / 2) + 0.5 * (y_top - y_bottom) / y_denom)
    if not (
        -(values.shape[1] - 1) / 2 < dx < (values.shape[1] - 1) / 2
        and -(values.shape[0] - 1) / 2 < dy < (values.shape[0] - 1) / 2
    ):
        return NccRefinement(False, None, None, peak, margin, "unstable_subpixel_peak")
    return NccRefinement(True, dx, dy, peak, margin, None)
