from lemonade_vision.pipeline.confidence import compute_confidence


def test_full_confidence_auto_add():
    result = compute_confidence(upc=1.0, vlm=1.0, embedding=1.0, dimension=1.0)
    assert result.final >= 0.85
    assert result.auto_add is True
    assert result.requires_verification is False


def test_partial_confidence_requires_verification():
    result = compute_confidence(upc=0.7, vlm=0.7, embedding=0.5, dimension=0.0)
    assert 0.50 <= result.final < 0.85
    assert result.auto_add is False
    assert result.requires_verification is True


def test_low_confidence_reject():
    result = compute_confidence(upc=0.0, vlm=0.1, embedding=0.1, dimension=0.0)
    assert result.final < 0.50
    assert result.auto_add is False
    assert result.requires_verification is False


def test_weights_sum_to_one():
    from lemonade_vision.pipeline.confidence import WEIGHTS
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6
