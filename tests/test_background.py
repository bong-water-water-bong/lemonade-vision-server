import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
import numpy as np

from lemonade_vision.pipeline.background import remove_background


def test_remove_background_produces_output():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        in_path = Path(d) / "input.jpg"
        out_path = Path(d) / "out.jpg"
        img = Image.fromarray(np.full((50, 50, 3), 128, dtype=np.uint8))
        img.save(str(in_path))

        result = remove_background(in_path, out_path)

        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_remove_background_with_rembg_available():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        in_path = Path(d) / "input.jpg"
        out_path = Path(d) / "out_bg.jpg"
        img = Image.fromarray(np.full((50, 50, 3), 128, dtype=np.uint8))
        img.save(str(in_path))

        with patch("rembg.remove", return_value=b"stripped"):
            result = remove_background(in_path, out_path)

        assert result == out_path
        assert out_path.exists()
        assert out_path.read_bytes() == b"stripped"


def test_remove_background_missing_input_returns_out_path():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        in_path = Path(d) / "nonexistent.jpg"
        out_path = Path(d) / "out.jpg"

        with pytest.raises(FileNotFoundError):
            remove_background(in_path, out_path)


@pytest.mark.asyncio
async def test_draft_assembler_includes_background_removal():
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult
    from unittest.mock import AsyncMock, MagicMock

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(
        brand="Test", vlm_status="ok", confidence=0.9
    ))
    assembler = DraftAssembler(vlm_client=vlm_client, embedding_model=None)

    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        still = str(Path(d) / "front.jpg")
        img.save(still)

        result = await assembler.run(
            job_id="j1", session_id="s1",
            rotation_video_path=None,
            still_paths={"front": still},
            depth_path=None,
            narration_path=None,
            frame_out_dir=d,
        )

        assert result["brand"] == "Test"
        assert still in result["frame_paths"]
        bg_dir = Path(d) / "bg"
        assert bg_dir.exists()


@pytest.mark.asyncio
async def test_bg_removal_still_only_nonexistent_frame_out_dir():
    """Still-only captures may not have frame_out_dir created yet — bg dir must still work."""
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult
    from unittest.mock import AsyncMock, MagicMock

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(
        brand="Test", vlm_status="ok", confidence=0.9
    ))
    assembler = DraftAssembler(vlm_client=vlm_client, embedding_model=None)

    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        still = str(Path(d) / "front.jpg")
        img.save(still)

        nested_out = str(Path(d) / "does_not_exist_yet" / "frames")
        result = await assembler.run(
            job_id="j2", session_id="s1",
            rotation_video_path=None,
            still_paths={"front": still},
            depth_path=None,
            narration_path=None,
            frame_out_dir=nested_out,
        )

        assert result["brand"] == "Test"
        bg_dir = Path(nested_out) / "bg"
        assert bg_dir.exists()
