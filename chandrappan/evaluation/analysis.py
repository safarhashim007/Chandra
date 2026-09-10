"""Deterministic correspondence error and confidence analyses."""

from __future__ import annotations

import math

import numpy as np


def error_distribution(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("error distribution requires at least one finite error")
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "fraction_above_5_px": float(np.mean(values > 5)),
        "fraction_above_10_px": float(np.mean(values > 10)),
        "fraction_above_25_px": float(np.mean(values > 25)),
        "fraction_above_50_px": float(np.mean(values > 50)),
    }


def confidence_subsets(
    errors: np.ndarray,
    confidence: np.ndarray,
    fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.1),
) -> list[dict[str, float | int]]:
    errors = np.asarray(errors, dtype=float).ravel()
    confidence = np.asarray(confidence, dtype=float).ravel()
    if len(errors) != len(confidence) or not len(errors):
        raise ValueError("errors and confidence must be non-empty and equally sized")
    valid = np.isfinite(errors) & np.isfinite(confidence)
    errors, confidence = errors[valid], confidence[valid]
    order = np.argsort(-confidence, kind="stable")
    result = []
    for fraction in fractions:
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("confidence fractions must be in (0, 1]")
        count = max(1, math.ceil(len(errors) * fraction))
        selected = errors[order[:count]]
        result.append(
            {
                "fraction": fraction,
                "count": int(count),
                "median_epe_px": float(np.median(selected)),
                "mean_epe_px": float(np.mean(selected)),
                "pck_1": float(np.mean(selected <= 1)),
                "pck_3": float(np.mean(selected <= 3)),
                "pck_5": float(np.mean(selected <= 5)),
            }
        )
    return result
