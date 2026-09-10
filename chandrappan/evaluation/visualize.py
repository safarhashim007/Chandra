"""Optional headless match diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_match_diagnostic(
    image_a: np.ndarray,
    image_b: np.ndarray,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "registration",
) -> None:
    import matplotlib.pyplot as plt

    first, second = np.asarray(image_a), np.asarray(image_b)
    source, target = np.asarray(source_xy), np.asarray(target_xy)
    inliers = np.asarray(inlier_mask, dtype=bool)
    if len(source) != len(target) or len(source) != len(inliers):
        raise ValueError("match arrays must have equal length")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(first, cmap="gray")
    axes[1].imshow(second, cmap="gray")
    for index in range(len(source)):
        color = "lime" if inliers[index] else "red"
        axes[0].plot(source[index, 0], source[index, 1], ".", color=color)
        axes[1].plot(target[index, 0], target[index, 1], ".", color=color)
    axes[0].set_title("source")
    axes[1].set_title("target")
    figure.suptitle(title)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)
