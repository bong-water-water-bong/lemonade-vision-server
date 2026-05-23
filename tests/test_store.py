import tempfile
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from lemonade_vision.store.schema import init_db
from lemonade_vision.store.product_db import ProductDB
from lemonade_vision.store.vector_db import VectorStore
from lemonade_vision.store.image_store import ImageStore


@pytest.fixture
def product_db():
    conn = init_db(":memory:")
    return ProductDB(conn)


@pytest.fixture
def vector_store():
    p = Path(tempfile.mkdtemp(dir="/tmp")) / "chroma"
    return VectorStore(str(p))


@pytest.fixture
def image_store():
    p = Path(tempfile.mkdtemp(dir="/tmp")) / "images"
    return ImageStore(str(p))


def test_product_db_insert_and_fetch(product_db):
    product_db.insert_product(
        sku="SKU001", brand="Elf Bar", flavor="Mango Ice",
        category="disposable_vape", puff_count=5000,
    )
    row = product_db.get_product("SKU001")
    assert row is not None
    assert row["brand"] == "Elf Bar"


def test_product_db_add_alias(product_db):
    product_db.insert_product(
        sku="SKU002", brand="Lost Mary", flavor="Watermelon",
        category="disposable_vape",
    )
    product_db.add_alias("SKU002", "blue one")
    aliases = product_db.get_aliases("SKU002")
    assert "blue one" in aliases


def test_vector_store_upsert_and_query(vector_store):
    vec = np.random.rand(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    vector_store.upsert_text("SKU001", vec, {"sku": "SKU001", "brand": "Elf Bar"})
    results = vector_store.query_text(vec, top_k=1)
    assert len(results) >= 1
    assert results[0]["id"] == "SKU001"


def test_image_store_save_returns_path(image_store):
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        src = Path(d) / "front.jpg"
        img = Image.fromarray(np.full((300, 300, 3), 128, dtype=np.uint8))
        img.save(str(src))
        out = image_store.save_image("SKU001", "front", src)
        assert out.exists()
        assert "SKU001" in str(out)
