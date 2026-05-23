from dataclasses import dataclass

WEIGHTS = {
    "upc": 0.40,
    "vlm": 0.30,
    "embedding": 0.20,
    "dimension": 0.10,
}

THRESHOLD_AUTO = 0.85
THRESHOLD_VERIFY = 0.50


@dataclass
class ConfidenceResult:
    final: float
    auto_add: bool
    requires_verification: bool


def compute_confidence(
    upc: float,
    vlm: float,
    embedding: float,
    dimension: float,
) -> ConfidenceResult:
    final = (
        WEIGHTS["upc"] * upc
        + WEIGHTS["vlm"] * vlm
        + WEIGHTS["embedding"] * embedding
        + WEIGHTS["dimension"] * dimension
    )
    return ConfidenceResult(
        final=round(final, 4),
        auto_add=final >= THRESHOLD_AUTO,
        requires_verification=THRESHOLD_VERIFY <= final < THRESHOLD_AUTO,
    )
