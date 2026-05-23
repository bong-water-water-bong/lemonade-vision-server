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
    pid = product_db.insert_product(
        {"brand": "Elf Bar", "flavor": "Mango Ice", "category": "disposable_vape",
         "puff_count": 5000}
    )
    assert pid is not None
    row = product_db.get_product(pid)
    assert row is not None
    assert row["brand"] == "Elf Bar"


def test_product_db_add_alias(product_db):
    pid = product_db.insert_product(
        {"brand": "Lost Mary", "flavor": "Watermelon", "category": "disposable_vape"}
    )
    product_db.add_alias(pid, "blue one")
    aliases = product_db.get_aliases(pid)
    assert "blue one" in aliases


def test_vector_store_upsert_and_query(vector_store):
    vec = np.random.rand(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    vector_store.upsert_text("pid001", vec, {"product_id": "pid001", "brand": "Elf Bar"})
    results = vector_store.query_text(vec, top_k=1)
    assert len(results) >= 1
    assert results[0]["id"] == "pid001"


def test_image_store_save_returns_url(image_store):
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        src = Path(d) / "front.jpg"
        img = Image.fromarray(np.full((300, 300, 3), 128, dtype=np.uint8))
        img.save(str(src))
        url = image_store.save_image("pid001", src, label="front")
        assert isinstance(url, str)
        assert url.startswith("/images/pid001/")


# --- New test cases ---

def test_product_db_update_product(product_db):
    pid = product_db.insert_product(
        {"brand": "Original", "flavor": "Mint", "category": "disposable_vape"}
    )
    product_db.update_product(pid, {"brand": "Updated"})
    row = product_db.get_product(pid)
    assert row is not None
    assert row["brand"] == "Updated"


def test_vector_store_upsert_visual_and_query(vector_store):
    vec = np.random.rand(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    vector_store.upsert_visual("pid002", vec, {"product_id": "pid002", "brand": "Lost Mary"})
    results = vector_store.query_visual(vec, top_k=1)
    assert len(results) >= 1
    assert results[0]["id"] == "pid002"


def test_image_store_get_and_list(image_store):
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        src = Path(d) / "main.jpg"
        img = Image.fromarray(np.full((200, 200, 3), 64, dtype=np.uint8))
        img.save(str(src))
        url = image_store.save_image("pid003", src, label="main")

        # get_image_url returns a string starting with /images/
        fetched = image_store.get_image_url("pid003", "main")
        assert isinstance(fetched, str)
        assert fetched.startswith("/images/")

        # list_images returns list of strings containing the url
        listing = image_store.list_images("pid003")
        assert isinstance(listing, list)
        assert all(isinstance(s, str) for s in listing)
        assert url in listing

        # nonexistent label returns None
        missing = image_store.get_image_url("pid003", "nonexistent")
        assert missing is None
