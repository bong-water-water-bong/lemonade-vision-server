# tests/test_vlm.py
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from lemonade_vision.pipeline.vlm import VLMClient, VLMResult


# --- unit tests (no network) ---


@pytest.fixture
def client():
    return VLMClient(base_url="http://localhost:8001")


def test_vlm_result_defaults():
    r = VLMResult()
    assert r.brand is None
    assert r.vlm_status == "ok"


@pytest.mark.asyncio
async def test_extract_product_info_parses_json(client):
    mock_response_text = json.dumps(
        {
            "brand": "Elf Bar",
            "flavor": "Mango Ice",
            "category": "disposable_vape",
            "puff_count": 5000,
            "nicotine_mg": 50,
            "ocr_text": "5000 puffs",
            "warnings": [],
            "confidence": 0.9,
        }
    )
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"choices": [{"message": {"content": mock_response_text}}]}
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 120, dtype=np.uint8))
        p = Path(d) / "test.jpg"
        img.save(str(p))
        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=fake_response):
            result = await client.extract_product_info([str(p)], narration=None)
    assert result.brand == "Elf Bar"
    assert result.puff_count == 5000
    assert result.vlm_status == "ok"


@pytest.mark.asyncio
async def test_extract_product_info_handles_vlm_timeout(client):
    import httpx

    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 120, dtype=np.uint8))
        p = Path(d) / "test.jpg"
        img.save(str(p))
        with patch.object(
            client._http, "post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("t")
        ):
            result = await client.extract_product_info([str(p)], narration=None)
    assert result.vlm_status == "unavailable"
    assert result.brand is None


@pytest.mark.asyncio
async def test_deduce_product_signals_returns_structured(client):
    mock_response_text = json.dumps(
        {
            "brand": "Lost Mary",
            "flavor": "Watermelon Ice",
            "size": "OS5000",
            "color": None,
            "category": "disposable_vape",
        }
    )
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"choices": [{"message": {"content": mock_response_text}}]}
    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=fake_response):
        result = await client.deduce_product_signals("lost mary watermelon ice os5000")
    assert result["brand"] == "Lost Mary"


# --- integration test (requires live :8001) ---


@pytest.mark.skipif(
    not os.getenv("VISION_INTEGRATION"),
    reason="requires VISION_INTEGRATION=1 and live VLM on :8001",
)
@pytest.mark.asyncio
async def test_vlm_integration_real_call():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 120, dtype=np.uint8))
        p = Path(d) / "test.jpg"
        img.save(str(p))
        client = VLMClient(base_url="http://localhost:8001")
        result = await client.extract_product_info([str(p)], narration="Test narration")
    assert result.vlm_status in ("ok", "unavailable")
