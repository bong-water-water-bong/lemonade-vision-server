ASR_BASE_URL = "http://localhost:8004"


def cosine_to_confidence(distance: float) -> float:
    return max(0.0, 1.0 - distance / 2.0)
