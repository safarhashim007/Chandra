from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app


def test_region_endpoint_exposes_required_coordinate_conventions(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "rp001.yaml").write_text(
        """region_id: RP-001
site_id: E000N0000
site_name: Test Site
center_latitude: 1.0
center_longitude: 2.0
longitude_convention: east_positive
latitude_type: planetocentric
status: TEST
products: []
""",
        encoding="utf-8",
    )
    client = TestClient(create_app(config_root=config_root, pack_root=tmp_path / "packs"))
    response = client.get("/api/frontend/regions/RP-001")
    assert response.status_code == 200
    assert response.json()["region"]["longitude_convention"] == "east_positive"
    assert client.get("/api/frontend/regions/RP-001/observations").status_code == 409
