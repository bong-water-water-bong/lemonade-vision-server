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


@pytest.mark.asyncio
async def test_embedding_lookup_with_hit():
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult
    import numpy as np

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(
        brand="Elf Bar", vlm_status="ok", confidence=0.9
    ))

    embed_model = MagicMock()
    embed_model.encode_image.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    vector_store = MagicMock()
    vector_store.query_visual.return_value = [{"id": "prod-1", "distance": 0.6}]

    assembler = DraftAssembler(
        vlm_client=vlm_client, embedding_model=embed_model, vector_store=vector_store,
    )

    import tempfile
    from pathlib import Path
    from PIL import Image

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

    embed_model.encode_image.assert_called_once_with(still)
    vector_store.query_visual.assert_called_once()
    assert result["signal_scores"]["embedding"] == 0.7
    assert result["brand"] == "Elf Bar"


@pytest.mark.asyncio
async def test_embedding_empty_hit_fallback():
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult
    import numpy as np

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(
        brand="Lost Mary", vlm_status="ok", confidence=0.88
    ))

    embed_model = MagicMock()
    embed_model.encode_image.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    vector_store = MagicMock()
    vector_store.query_visual.return_value = []

    assembler = DraftAssembler(
        vlm_client=vlm_client, embedding_model=embed_model, vector_store=vector_store,
    )

    import tempfile
    from pathlib import Path
    from PIL import Image

    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        still = str(Path(d) / "front.jpg")
        img.save(still)

        result = await assembler.run(
            job_id="j2", session_id="s1",
            rotation_video_path=None,
            still_paths={"front": still},
            depth_path=None,
            narration_path=None,
            frame_out_dir=d,
        )

    assert result["signal_scores"]["embedding"] == 0.5


@pytest.mark.asyncio
async def test_embedding_distance_scoring():
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult
    import numpy as np

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(
        brand="Geek Bar", vlm_status="ok", confidence=0.85
    ))

    embed_model = MagicMock()
    embed_model.encode_image.return_value = np.array([0.5, 0.5], dtype=np.float32)

    vector_store = MagicMock()
    vector_store.query_visual.return_value = [{"id": "prod-2", "distance": 0.0}]

    assembler = DraftAssembler(
        vlm_client=vlm_client, embedding_model=embed_model, vector_store=vector_store,
    )

    import tempfile
    from pathlib import Path
    from PIL import Image

    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        still = str(Path(d) / "front.jpg")
        img.save(still)

        result = await assembler.run(
            job_id="j3", session_id="s1",
            rotation_video_path=None,
            still_paths={"front": still},
            depth_path=None,
            narration_path=None,
            frame_out_dir=d,
        )

    assert result["signal_scores"]["embedding"] == 1.0


@pytest.mark.asyncio
async def test_embedding_exception_fallback():
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult
    import numpy as np

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(
        brand="Raz", vlm_status="ok", confidence=0.8
    ))

    embed_model = MagicMock()
    embed_model.encode_image.side_effect = RuntimeError("CLIP crashed")

    vector_store = MagicMock()

    assembler = DraftAssembler(
        vlm_client=vlm_client, embedding_model=embed_model, vector_store=vector_store,
    )

    import tempfile
    from pathlib import Path
    from PIL import Image

    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        still = str(Path(d) / "front.jpg")
        img.save(still)

        result = await assembler.run(
            job_id="j4", session_id="s1",
            rotation_video_path=None,
            still_paths={"front": still},
            depth_path=None,
            narration_path=None,
            frame_out_dir=d,
        )

    assert result["signal_scores"]["embedding"] == 0.5
    assert result["brand"] == "Raz"


@pytest.mark.asyncio
async def test_embedding_skipped_when_no_vector_store():
    from lemonade_vision.draft import DraftAssembler
    from lemonade_vision.pipeline.vlm import VLMResult

    vlm_client = MagicMock()
    vlm_client.extract_product_info = AsyncMock(return_value=VLMResult(
        brand="Flum", vlm_status="ok", confidence=0.9
    ))

    embed_model = MagicMock()

    assembler = DraftAssembler(
        vlm_client=vlm_client, embedding_model=embed_model, vector_store=None,
    )

    import tempfile
    from pathlib import Path
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        still = str(Path(d) / "front.jpg")
        img.save(still)

        result = await assembler.run(
            job_id="j5", session_id="s1",
            rotation_video_path=None,
            still_paths={"front": still},
            depth_path=None,
            narration_path=None,
            frame_out_dir=d,
        )

    embed_model.encode_image.assert_not_called()
    assert result["signal_scores"]["embedding"] == 0.5
