# Confidence Scoring

> The confidence scorer combines four pipeline signal scores into a single weighted float and classifies the result as auto-add, requires-verification, or reject.

## Overview

`compute_confidence()` in `pipeline/confidence.py` takes four scalar scores (each 0.0–1.0) produced by the pipeline stages and returns a `ConfidenceResult` dataclass with the weighted final score and two boolean flags. The flags drive the cashier's disposition: `auto_add=True` means the product can be committed to inventory without human review; `requires_verification=True` means a human must confirm before commit; neither flag True means the scan should be rejected. The scorer has no external dependencies — it is a pure function operating on pre-computed floats.

## How It Works

The four input scores and their weights:

| Signal | Weight | Source |
|--------|--------|--------|
| `upc` | 0.40 | 1.0 if barcode decoded, else 0.0 |
| `vlm` | 0.30 | model's self-reported confidence (0–1), or 0.0 if VLM unavailable |
| `embedding` | 0.20 | currently 0.5 if any frames present, else 0.0 (see Gotchas) |
| `dimension` | 0.10 | 0.5 if depth grid parsed successfully, else 0.0 |

The final score is a simple weighted sum:

```
final = 0.40 * upc + 0.30 * vlm + 0.20 * embedding + 0.10 * dimension
```

Rounded to 4 decimal places.

**Threshold logic:**

```
THRESHOLD_AUTO   = 0.85  → auto_add = True if final >= 0.85
THRESHOLD_VERIFY = 0.50  → requires_verification = True if 0.50 <= final < 0.85
                           (neither flag) if final < 0.50 → reject
```

`auto_add` and `requires_verification` are mutually exclusive by construction: a score of exactly 0.85 sets `auto_add=True` and `requires_verification=False`. A score below 0.50 sets both to False.

**Worked examples:**

- UPC found, VLM confident (0.9), frames present, depth present:
  `0.40*1.0 + 0.30*0.9 + 0.20*0.5 + 0.10*0.5 = 0.40 + 0.27 + 0.10 + 0.05 = 0.82` → verify
- UPC found, VLM confident (0.95), frames present, depth present:
  `0.40 + 0.285 + 0.10 + 0.05 = 0.835` → verify (just below 0.85 auto)
- UPC found (0.40), VLM highly confident (1.0 → 0.30), frames + depth present:
  `0.40 + 0.30 + 0.10 + 0.05 = 0.85` → auto
- UPC missing, VLM confident (0.8), frames present, no depth:
  `0.0 + 0.24 + 0.10 + 0.0 = 0.34` → reject

## Key Decisions

- **UPC weighted highest at 40%**: A successfully decoded barcode is a deterministic, human-assigned product identifier. It eliminates ambiguity in a way that probabilistic visual signals cannot. When a UPC is present, it should dominate the score.

- **Hard threshold constants rather than per-product configuration**: `THRESHOLD_AUTO` and `THRESHOLD_VERIFY` are module-level constants. The `CommitRequest` model has a `confidence_threshold` field (default 0.85), but `compute_confidence` itself does not read it — the commit threshold is enforced at the product record level (the minimum confidence required for auto-commit on subsequent scans), not at draft-scoring time. This separates onboarding confidence from ongoing recognition confidence.

- **`ConfidenceResult` is a dataclass, not a dict**: Returning a typed dataclass forces callers to be explicit about which field they're reading (`result.auto_add` vs `result["auto_add"]`). The `final` score, `auto_add`, and `requires_verification` are all present in one object, preventing partial use.

## Gotchas

- **Embedding score is a placeholder, not a real similarity measure**: `assemble_draft()` sets `embedding_score = 0.5 if frame_paths else 0.0`. This means any scan with at least one frame always contributes a fixed 0.10 (0.20 weight × 0.5 score) to the final, regardless of whether any similar product exists in ChromaDB. For first-time onboarding this is reasonable (no prior embeddings to compare against), but for re-scans of known products the 0.5 constant understates actual similarity and the real CLIP distance is never used during scoring.

- **No UPC confidence gradation**: The `upc` score is binary (1.0 or 0.0). A partial barcode decode, a checksum mismatch, or a barcode type mismatch with the database all produce 0.0. There is no intermediate confidence for "likely UPC but checksum failed."

- **Score of exactly 0.85 is `auto_add=True`**: The boundary condition `final >= THRESHOLD_AUTO` means a score of exactly 0.85 does not go to verify — it auto-adds. Given the 4-decimal rounding, the effective floor for auto-add is `0.8500`. Callers should be aware that scores around this boundary can toggle on small VLM confidence changes.

## Related

- [[architecture]] — where `compute_confidence` is called in the overall flow
- [[scan-pipeline]] — the stages that produce the four input scores
