import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from lemonade_vision.pipeline.barcode import extract_upc


def test_extract_upc_no_barcode_returns_none():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((200, 200, 3), 200, dtype=np.uint8))
        p = Path(d) / "blank.jpg"
        img.save(str(p))
        assert extract_upc(p) is None


def test_extract_upc_missing_file_returns_none():
    assert extract_upc(Path("/tmp/no_such_file.jpg")) is None
