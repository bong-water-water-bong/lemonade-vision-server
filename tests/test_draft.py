# tests/test_draft.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from lemonade_vision.draft import assemble_draft


def test_assemble_draft_with_all_signals():
    from lemonade_vision.pipeline.vlm import VLMResult
    result = assemble_draft(
        job_id="j1",
        session_id="s1",
        vlm_result=VLMResult(
            brand="Elf Bar", flavor="Mango Ice",
            category="disposable_vape", puff_count=5000,
            nicotine_mg=50, ocr_text="5000 puffs",
            vlm_status="ok", confidence=0.92,
        ),
        upc="012345678901",
        dimensions=(25.0, 120.0, 25.0),
        narration="elf bar mango 5000",
        frame_paths=["frame1.jpg"],
    )
    assert result["brand"] == "Elf Bar"
    assert result["upc"] == "012345678901"
    assert result["signal_scores"]["upc"] > 0
    assert result["signal_scores"]["vlm"] > 0
    assert result["signal_scores"]["embedding"] == 0.5
    assert result["signal_scores"]["dimension"] == 0.5


def test_assemble_draft_missing_barcode_still_ok():
    from lemonade_vision.pipeline.vlm import VLMResult
    result = assemble_draft(
        job_id="j2", session_id="s1",
        vlm_result=VLMResult(brand="Lost Mary", flavor="Watermelon", vlm_status="ok"),
        upc=None,
        dimensions=None,
        narration=None,
        frame_paths=[],
    )
    assert result["upc"] is None
    assert result["signal_scores"]["upc"] == 0.0


def test_assemble_draft_vlm_unavailable():
    from lemonade_vision.pipeline.vlm import VLMResult
    result = assemble_draft(
        job_id="j3", session_id="s1",
        vlm_result=VLMResult(vlm_status="unavailable"),
        upc=None, dimensions=None, narration=None, frame_paths=[],
    )
    assert result["vlm_status"] == "unavailable"
    assert result["brand"] is None


@pytest.mark.asyncio
async def test_draft_assembler_run_with_stills():
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(
        brand="Elf Bar", vlm_status="ok", confidence=0.9
    ))
    assembler = DraftAssembler(vlm_client=vlm_client, embedding_model=None)

    import tempfile
    from pathlib import Path
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        # Create a dummy still
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        still = str(Path(d) / "upc.jpg")
        img.save(still)

        result = await assembler.run(
            job_id="j1", session_id="s1",
            rotation_video_path=None,
            still_paths={"upc": still},
            depth_path=None,
            narration_path=None,
            frame_out_dir=d,
        )

    assert result["brand"] == "Elf Bar"
    assert still in result["frame_paths"]


@pytest.mark.asyncio
async def test_draft_assembler_run_resilient_to_bad_video():
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(vlm_status="unavailable"))
    assembler = DraftAssembler(vlm_client=vlm_client, embedding_model=None)

    import tempfile
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        result = await assembler.run(
            job_id="j2", session_id="s1",
            rotation_video_path="/tmp/nonexistent_video.mp4",
            still_paths={},
            depth_path=None,
            narration_path=None,
            frame_out_dir=d,
        )

    assert result["vlm_status"] == "unavailable"
    assert result["frame_paths"] == []


@pytest.mark.asyncio
async def test_draft_assembler_no_duplicate_paths():
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(vlm_status="ok"))
    assembler = DraftAssembler(vlm_client=vlm_client, embedding_model=None)

    import tempfile
    from pathlib import Path
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        still = str(Path(d) / "front.jpg")
        img.save(still)

        result = await assembler.run(
            job_id="j3", session_id="s1",
            rotation_video_path=None,
            still_paths={"front": still, "back": still},  # same path twice
            depth_path=None,
            narration_path=None,
            frame_out_dir=d,
        )

    assert result["frame_paths"].count(still) == 1
