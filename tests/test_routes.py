# tests/test_routes.py
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    import os

    os.environ["VISION_DATA_DIR"] = str(tmp_path / "data")
    from lemonade_vision.server import create_app

    app = create_app(data_dir=str(tmp_path / "data"))
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "vlm_reachable" in data


def test_session_start_returns_session_id(client):
    resp = client.post("/session/start")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "qr_png_b64" in data


def test_capture_still_requires_session_token(client):
    resp = client.post(
        "/capture/still",
        data={"angle": "front"},
        files={"file": ("test.jpg", b"data", "image/jpeg")},
    )
    assert resp.status_code == 401


def test_capture_still_bad_angle_returns_422(client):
    sess_resp = client.post("/session/start")
    sid = sess_resp.json()["session_id"]
    resp = client.post(
        "/capture/still",
        headers={"X-Session-Token": sid},
        data={"angle": "invalid_angle"},
        files={"file": ("test.jpg", b"\xff\xd8\xff", "image/jpeg")},
    )
    assert resp.status_code == 422


def test_session_delete_returns_204(client):
    sess_resp = client.post("/session/start")
    sid = sess_resp.json()["session_id"]
    resp = client.delete(f"/session/{sid}")
    assert resp.status_code == 204


def test_product_draft_unknown_job_returns_404(client):
    resp = client.get("/product/draft/nonexistent-job-id")
    assert resp.status_code == 404
