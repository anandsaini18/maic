"""Tests for frontend static file mounting and root route behavior."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from main import configure_frontend


def test_root_and_assets_served_when_frontend_build_exists(tmp_path):
    static_dir = tmp_path / "static"
    assets_dir = static_dir / "assets"
    assets_dir.mkdir(parents=True)

    (static_dir / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('ok');", encoding="utf-8")

    app = FastAPI()
    configure_frontend(app, static_dir)
    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "ok" in root_resp.text

    asset_resp = client.get("/assets/app.js")
    assert asset_resp.status_code == 200
    assert "console.log('ok');" in asset_resp.text


def test_root_returns_503_when_assets_are_missing(tmp_path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>stale</body></html>", encoding="utf-8")

    app = FastAPI()
    configure_frontend(app, static_dir)
    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == 503
    assert "just build" in root_resp.text
    assert client.get("/assets/does-not-exist.js").status_code == 404
