"""Local-only Chandrappan API and offline RP-001 presentation frontend."""

from __future__ import annotations

import csv
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chandrappan.demo_runtime import DemoRuntime


def _regions(config_root: Path) -> dict[str, dict[str, object]]:
    regions = {}
    for path in sorted(config_root.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload and payload.get("region_id"):
            regions[str(payload["region_id"])] = payload
    return regions


def _region_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "id": payload["region_id"],
        "site_id": payload["site_id"],
        "site_name": payload["site_name"],
        "latitude": payload["center_latitude"],
        "longitude": payload["center_longitude"],
        "longitude_convention": payload["longitude_convention"],
        "latitude_type": payload["latitude_type"],
        "status": payload.get("status"),
        "product_count": len(payload.get("products", [])),
    }


def create_app(
    *,
    config_root: Path = Path("configs/regions"),
    pack_root: Path = Path("data/region_packs"),
    demo_root: Path | None = None,
    enable_demo: bool = False,
    device: str = "auto",
) -> FastAPI:
    """Create metadata endpoints, optionally with the offline live demo.

    Tests and metadata-only uses leave ``enable_demo`` false, preventing model
    allocation. The Mac start script enables it and waits for model preload.
    """
    runtime: DemoRuntime | None = None
    if enable_demo:
        if demo_root is None:
            raise ValueError("demo_root is required when the live demo is enabled")
        runtime = DemoRuntime(demo_root, device=device)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            runtime.preload()
        yield

    app = FastAPI(title="Chandrappan", version="0.1", lifespan=lifespan)

    @app.get("/api/frontend/regions")
    def list_regions() -> dict[str, list[dict[str, object]]]:
        return {"regions": [_region_summary(row) for row in _regions(config_root).values()]}

    @app.get("/api/frontend/regions/{region_id}")
    def get_region(region_id: str) -> dict[str, object]:
        try:
            payload = _regions(config_root)[region_id]
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown region pack: {region_id}"
            ) from exc
        return {"region": _region_summary(payload), "products": payload["products"]}

    @app.get("/api/frontend/regions/{region_id}/observations")
    def get_observations(region_id: str) -> dict[str, object]:
        if region_id not in _regions(config_root):
            raise HTTPException(status_code=404, detail=f"unknown region pack: {region_id}")
        catalog = pack_root / region_id / "observation_catalog.csv"
        if not catalog.exists():
            raise HTTPException(status_code=409, detail=f"region pack not yet built: {region_id}")
        with catalog.open(newline="", encoding="utf-8") as handle:
            observations = list(csv.DictReader(handle))
        return {"region_id": region_id, "observations": observations}

    def require_runtime() -> DemoRuntime:
        if runtime is None:
            raise HTTPException(status_code=503, detail="live demo profile is disabled")
        return runtime

    @app.get("/api/demo/health")
    def demo_health() -> dict[str, object]:
        return require_runtime().status()

    @app.get("/api/demo/summary")
    def demo_summary() -> dict[str, object]:
        active_runtime = require_runtime()
        if not active_runtime.manifest:
            raise HTTPException(status_code=503, detail="model preload in progress")
        return {
            "region_id": "RP-001",
            "site_name": "Apollo 15 S-IVB Impact",
            "model": "Official RoMa v2",
            "d1": "EXPERIMENTAL · NOT PROMOTED",
            "metrics": active_runtime.manifest["presentation_metrics"],
            "scientific_details": active_runtime.manifest["scientific_details"],
            "known_cases": active_runtime.manifest["known_cases"],
        }

    @app.get("/api/demo/reference-bank")
    def demo_reference_bank() -> dict[str, object]:
        return {"observations": require_runtime().bank_summary()}

    @app.post("/api/demo/run/{case_name}")
    def run_demo_case(case_name: str) -> dict[str, object]:
        try:
            return require_runtime().run_case(case_name)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    if runtime is not None:
        runtime.paths.artifacts.mkdir(parents=True, exist_ok=True)
        app.mount("/artifacts", StaticFiles(directory=runtime.paths.artifacts), name="artifacts")
        app.mount("/demo-data", StaticFiles(directory=runtime.paths.data_root), name="demo-data")

        @app.get("/")
        def demo_frontend() -> FileResponse:
            return FileResponse(runtime.paths.frontend / "index.html")

    return app


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEMO_ROOT = Path(os.environ.get("CHANDRAPPAN_DEMO_ROOT", _PROJECT_ROOT))
app = create_app(
    config_root=_PROJECT_ROOT / "configs" / "regions",
    pack_root=_PROJECT_ROOT / "data" / "region_packs",
    demo_root=_DEMO_ROOT,
    enable_demo=(_DEMO_ROOT / "data" / "rp001" / "demo_manifest.json").is_file(),
    device=os.environ.get("CHANDRAPPAN_DEVICE", "auto"),
)
