# src/lemonade_vision/api/deduce.py
from __future__ import annotations
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from lemonade_vision.models import DeduceCandidate, DeduceRequest, DeduceResponse

router = APIRouter()

ALIAS_BONUS = 0.15
BRAND_BONUS = 0.10
FLAVOR_BONUS = 0.10

_logger = logging.getLogger(__name__)


def _cosine_to_confidence(distance: float) -> float:
    return max(0.0, 1.0 - distance / 2.0)


@router.post("/deduce/text", response_model=DeduceResponse)
async def deduce_text(body: DeduceRequest, request: Request):
    vlm_client = request.app.state.vlm_client
    embed_model = request.app.state.embed_model
    vector_store = request.app.state.vector_store
    product_db = request.app.state.product_db

    signals = await vlm_client.deduce_product_signals(body.query)
    brand_hint: Optional[str] = signals.get("brand")
    flavor_hint: Optional[str] = signals.get("flavor")

    enriched = " ".join(filter(None, [
        brand_hint, flavor_hint,
        signals.get("size"), signals.get("category"),
        body.query,
    ]))
    query_vec = embed_model.encode_text(enriched)

    raw = vector_store.query_text(query_vec, top_k=body.top_k * 2)

    candidates: list[DeduceCandidate] = []
    for hit in raw:
        meta = hit["metadata"]
        sku = hit["id"]
        confidence = _cosine_to_confidence(hit["distance"])

        reasons: list[str] = []
        if brand_hint and meta.get("brand", "").lower() == brand_hint.lower():
            confidence = min(1.0, confidence + BRAND_BONUS)
            reasons.append("brand match")
        if flavor_hint and meta.get("flavor", "").lower() == flavor_hint.lower():
            confidence = min(1.0, confidence + FLAVOR_BONUS)
            reasons.append("flavor match")

        aliases = product_db.get_aliases(sku)
        query_lower = body.query.lower()
        if any(a.lower() in query_lower for a in aliases):
            confidence = min(1.0, confidence + ALIAS_BONUS)
            reasons.append("alias match")

        candidates.append(DeduceCandidate(
            sku=sku,
            confidence=round(confidence, 4),
            match_reason=", ".join(reasons) or "embedding similarity",
            brand=meta.get("brand"),
            flavor=meta.get("flavor"),
        ))

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return DeduceResponse(candidates=candidates[:body.top_k], query_used=enriched)


@router.post("/deduce/audio", response_model=DeduceResponse)
async def deduce_audio(request: Request, file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        suffix = Path(file.filename or "query.bin").suffix or ".bin"
        audio_path = Path(d) / f"query{suffix}"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        wav_path = Path(d) / "query.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path),
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=500, detail=f"ffmpeg conversion failed: {exc}")

        transcript: Optional[str] = None
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                with open(wav_path, "rb") as af:
                    resp = await client.post(
                        "http://localhost:8004/transcribe",
                        content=af.read(),
                        headers={"Content-Type": "audio/wav"},
                    )
                    resp.raise_for_status()
                    transcript = resp.json().get("text")
        except Exception as exc:
            _logger.warning("fw-server transcription failed: %s", exc)
            raise HTTPException(status_code=503, detail="fw-server :8004 unreachable")

        if not transcript:
            raise HTTPException(status_code=503, detail="transcription returned empty")

        return await deduce_text(body=DeduceRequest(query=transcript), request=request)
