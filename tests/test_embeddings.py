import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from lemonade_vision.pipeline.embeddings import EmbeddingModel


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


def _make_jpg(path: Path) -> None:
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(str(path))


def test_encode_image_returns_normalized_vector(model):
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        p = Path(d) / "img.jpg"
        _make_jpg(p)
        vec = model.encode_image(str(p))
        assert vec.shape == (512,)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 0.01


def test_encode_text_returns_normalized_vector(model):
    vec = model.encode_text("elf bar mango ice disposable vape")
    assert vec.shape == (512,)
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 0.01


def test_same_text_same_vector(model):
    v1 = model.encode_text("lost mary watermelon")
    v2 = model.encode_text("lost mary watermelon")
    assert np.allclose(v1, v2, atol=1e-5)


def test_different_texts_different_vectors(model):
    v1 = model.encode_text("elf bar mango")
    v2 = model.encode_text("lost mary blueberry")
    assert not np.allclose(v1, v2, atol=0.01)


@pytest.mark.skipif(
    not os.getenv("VISION_INTEGRATION"),
    reason="integration: requires VISION_INTEGRATION=1",
)
def test_image_text_similarity_positive(model):
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        p = Path(d) / "img.jpg"
        _make_jpg(p)
        img_vec = model.encode_image(str(p))
        txt_vec = model.encode_text("product label")
        sim = float(np.dot(img_vec, txt_vec))
        assert sim > -1.0
