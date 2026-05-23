# tests/test_pipeline_integration.py
"""
End-to-end pipeline integration tests.
Requires: VISION_INTEGRATION=1 and live VLM on :8001.
Skipped automatically otherwise.
"""
import os
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

pytestmark = pytest.mark.skipif(
    not os.getenv("VISION_INTEGRATION"),
    reason="requires VISION_INTEGRATION=1 and live services",
)


@pytest.fixture
def sample_image():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        p = Path(d) / "product.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img.save(str(p))
        yield str(p)


@pytest.fixture
def sample_depth():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        p = Path(d) / "depth.json"
        grid = np.full((256, 192), 350.0).tolist()
        p.write_text(json.dumps(grid))
        yield str(p)


@pytest.mark.asyncio
async def test_vlm_client_real_call(sample_image):
    from lemonade_vision.pipeline.vlm import VLMClient
    client = VLMClient()
    result = await client.extract_product_info([sample_image], narration=None)
    assert result.vlm_status in ("ok", "unavailable")


@pytest.mark.asyncio
async def test_depth_to_dimensions_from_fixture():
    from lemonade_vision.pipeline.dimensions import depth_to_dimensions
    fixture = Path("tests/fixtures/depth-sample.json")
    grid = np.array(json.loads(fixture.read_text()))
    dims = depth_to_dimensions(grid)
    assert dims is not None
    w, h, d = dims
    assert 0 < w < 2000
    assert 0 < h < 2000


@pytest.mark.asyncio
async def test_full_server_health():
    from fastapi.testclient import TestClient
    from lemonade_vision.server import create_app
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        app = create_app(data_dir=d)
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
