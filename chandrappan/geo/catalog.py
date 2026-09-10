"""Local SQLite + RTree lunar image catalog."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .metadata import LunarImageMetadata


class LunarCatalog:
    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY,
                product_id TEXT UNIQUE NOT NULL,
                source_path TEXT NOT NULL,
                center_lat REAL NOT NULL,
                center_lon_east REAL NOT NULL,
                gsd_m_per_px REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS observation_bounds USING rtree(
                id, min_x, max_x, min_y, max_y
            );
            """
        )

    def add(self, metadata: LunarImageMetadata) -> None:
        corners = [
            metadata.transform.pixel_to_world(x, y)
            for x, y in (
                (0, 0),
                (metadata.width, 0),
                (metadata.width, metadata.height),
                (0, metadata.height),
            )
        ]
        xs, ys = [float(c[0]) for c in corners], [float(c[1]) for c in corners]
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO observations("
                "product_id, source_path, center_lat, center_lon_east, gsd_m_per_px"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    metadata.product_id,
                    metadata.source_path,
                    metadata.center_lat,
                    metadata.center_lon_east,
                    metadata.gsd_m_per_px,
                ),
            )
            row = self.connection.execute(
                "SELECT id FROM observations WHERE product_id = ?", (metadata.product_id,)
            ).fetchone()
            assert row is not None
            self.connection.execute("DELETE FROM observation_bounds WHERE id = ?", (row["id"],))
            self.connection.execute(
                "INSERT INTO observation_bounds VALUES (?, ?, ?, ?, ?)",
                (row["id"], min(xs), max(xs), min(ys), max(ys)),
            )

    def query_bounds(
        self, min_x: float, max_x: float, min_y: float, max_y: float
    ) -> list[sqlite3.Row]:
        return self.connection.execute(
            """SELECT o.* FROM observation_bounds b JOIN observations o ON b.id=o.id
            WHERE b.max_x >= ? AND b.min_x <= ? AND b.max_y >= ? AND b.min_y <= ?""",
            (min_x, max_x, min_y, max_y),
        ).fetchall()

    def close(self) -> None:
        self.connection.close()
