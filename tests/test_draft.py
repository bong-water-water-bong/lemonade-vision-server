# tests/test_draft.py
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
