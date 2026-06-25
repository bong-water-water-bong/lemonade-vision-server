from lemonade_vision.models import (
    SessionStartResponse,
    SignalScores,
    DraftProduct,
    DeduceCandidate,
    DeduceResponse,
)


def test_session_start_response_has_session_id_and_qr():
    r = SessionStartResponse(session_id="abc", qr_png_b64="data")
    assert r.session_id == "abc"
    assert r.qr_png_b64 == "data"


def test_signal_scores_clamp_to_float():
    s = SignalScores(upc=1.0, vlm=0.9, embedding=0.8, dimension=0.7)
    assert s.upc == 1.0


def test_draft_product_optional_fields_default_none():
    d = DraftProduct(job_id="j1", status="ready")
    assert d.upc is None
    assert d.brand is None
    assert d.dimensions is None


def test_deduce_response_has_candidates():
    c = DeduceCandidate(sku="SKU001", confidence=0.9, match_reason="brand+flavor")
    r = DeduceResponse(candidates=[c])
    assert len(r.candidates) == 1
    assert r.candidates[0].sku == "SKU001"
