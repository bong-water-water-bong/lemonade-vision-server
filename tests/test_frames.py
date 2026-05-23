import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from lemonade_vision.pipeline.frames import (
    SECTORS,
    laplacian_variance,
    select_sharpest_frames,
)


def _make_sharp_jpg(path: Path) -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:110, 200:440] = 255  # high-contrast edge
    Image.fromarray(img).save(str(path), "JPEG")


def _make_blurry_jpg(path: Path) -> None:
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    Image.fromarray(img).save(str(path), "JPEG")


def test_laplacian_variance_sharp_greater_than_blurry():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        sharp = Path(d) / "sharp.jpg"
        blurry = Path(d) / "blurry.jpg"
        _make_sharp_jpg(sharp)
        _make_blurry_jpg(blurry)
        assert laplacian_variance(sharp) > laplacian_variance(blurry)


def test_select_sharpest_frames_returns_one_per_sector():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        frames = []
        for i in range(SECTORS * 2):
            p = Path(d) / f"frame_{i:03d}.jpg"
            _make_sharp_jpg(p)
            frames.append((i * (360 // (SECTORS * 2)), str(p)))
        selected = select_sharpest_frames(frames)
        assert len(selected) == SECTORS


def test_laplacian_variance_missing_file_returns_zero():
    assert laplacian_variance(Path("/tmp/does_not_exist.jpg")) == 0.0
