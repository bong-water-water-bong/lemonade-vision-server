# tests/test_deduce.py
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_products(tmp_path):
    import os
    os.environ["VISION_DATA_DIR"] = str(tmp_path / "data")
    from lemonade_vision.server import create_app
    app = create_app(data_dir=str(tmp_path / "data"))

    with TestClient(app) as client:
        product_db = app.state.product_db
        vector_store = app.state.vector_store
        embed_model = app.state.embed_model

        product_db.insert_product({
            "product_id": "ELFBAR001",
            "brand": "Elf Bar", "flavor": "Mango Ice",
            "category": "disposable_vape", "puff_count": 5000,
        })
        vec = embed_model.encode_text("Elf Bar Mango Ice disposable_vape")
        vector_store.upsert_text("ELFBAR001", vec, {
            "sku": "ELFBAR001", "brand": "Elf Bar",
            "flavor": "Mango Ice", "category": "disposable_vape",
        })
        yield client


def test_deduce_text_returns_candidates(client_with_products):
    with patch(
        "lemonade_vision.pipeline.vlm.VLMClient.deduce_product_signals",
        new_callable=AsyncMock,
        return_value={"brand": "Elf Bar", "flavor": "Mango Ice", "category": "disposable_vape"},
    ):
        resp = client_with_products.post(
            "/deduce/text",
            json={"query": "elf bar mango ice 5000", "top_k": 3},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data


def test_deduce_text_missing_query_returns_422(client_with_products):
    resp = client_with_products.post("/deduce/text", json={"top_k": 3})
    assert resp.status_code == 422


def test_deduce_audio_no_file_returns_422(client_with_products):
    resp = client_with_products.post("/deduce/audio")
    assert resp.status_code == 422
