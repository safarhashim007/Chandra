"""Explicit affine pixel/world transforms with unambiguous x/y ordering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineTransform:
    """Maps pixel (x, y) to world coordinates with GDAL-style six coefficients."""

    x_origin: float
    x_col: float
    x_row: float
    y_origin: float
    y_col: float
    y_row: float

    def matrix(self) -> np.ndarray:
        return np.array([[self.x_col, self.x_row], [self.y_col, self.y_row]], dtype=np.float64)

    def pixel_to_world(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return (
            self.x_origin + self.x_col * x_arr + self.x_row * y_arr,
            self.y_origin + self.y_col * x_arr + self.y_row * y_arr,
        )

    def world_to_pixel(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.matrix()
        if abs(np.linalg.det(matrix)) < 1e-15:
            raise ValueError("Pixel/world affine transform is singular")
        coords = np.stack(
            (
                np.asarray(x, dtype=float) - self.x_origin,
                np.asarray(y, dtype=float) - self.y_origin,
            ),
            axis=0,
        )
        pixels = np.linalg.solve(matrix, coords.reshape(2, -1)).reshape(coords.shape)
        return pixels[0], pixels[1]
