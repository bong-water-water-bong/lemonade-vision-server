import subprocess
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def authed_client(tmp_path):
    import os

    os.environ["VISION_DATA_DIR"] = str(tmp_path / "data")
    from lemonade_vision.server import create_app

    app = create_app(data_dir=str(tmp_path / "data"))
    with TestClient(app) as client:
        resp = client.post("/session/start")
        assert resp.status_code == 200
        token = resp.json()["session_id"]
        yield client, token


def test_capture_audio_calls_ffmpeg_for_m4a(authed_client):
    """Narration M4A upload triggers ffmpeg conversion to WAV."""
    client, token = authed_client
    with patch("lemonade_vision.api.capture.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        resp = client.post(
            "/capture/audio",
            headers={"X-Session-Token": token},
            files={"file": ("narration.m4a", b"\x00" * 16, "audio/m4a")},
        )
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "pcm_s16le" in cmd
    assert resp.status_code == 500
