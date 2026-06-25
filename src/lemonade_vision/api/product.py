# src/lemonade_vision/api/product.py
from __future__ import annotations
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from lemonade_vision.models import (
    CommitRequest,
    CommitResponse,
    DraftProduct,
    ProductPatch,
    SignalScores,
)

_logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/product/draft/{job_id}", response_model=DraftProduct)
async def get_draft(job_id: str, request: Request):
    db = request.app.state.db
    row = db.execute("SELECT * FROM draft_jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    draft = json.loads(row["draft_json"]) if row["draft_json"] else {}
    scores = SignalScores(**json.loads(row["signal_scores"])) if row["signal_scores"] else None
    return DraftProduct(
        job_id=job_id,
        status=row["status"],
        signal_scores=scores,
        **{
            k: v
            for k, v in draft.items()
            if k in DraftProduct.model_fields and k not in ("job_id", "status", "signal_scores")
        },
    )


@router.post("/product/commit", response_model=CommitResponse)
async def commit_product(body: CommitRequest, request: Request):
    db = request.app.state.db
    product_db = request.app.state.product_db
    vector_store = request.app.state.vector_store
    image_store = request.app.state.image_store
    embed_model = request.app.state.embed_model

    row = db.execute("SELECT * FROM draft_jobs WHERE job_id = ?", (body.job_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="draft job not found")

    draft = json.loads(row["draft_json"]) if row["draft_json"] else {}

    product_db.insert_product(
        {
            "product_id": body.sku,
            "brand": body.brand,
            "flavor": body.flavor,
            "category": body.category,
            "barcode": draft.get("upc"),
            "puff_count": body.puff_count,
            "nicotine_mg": body.nicotine_mg,
            "ocr_text": draft.get("ocr_text"),
            "confidence": body.confidence_threshold,
            "requires_attendant": body.requires_attendant,
        }
    )

    for alias in body.aliases:
        product_db.add_alias(body.sku, alias)

    text_query = f"{body.brand} {body.flavor} {body.category} {' '.join(body.aliases)}"
    text_vec = embed_model.encode_text(text_query)
    vector_store.upsert_text(
        body.sku,
        text_vec,
        {
            "sku": body.sku,
            "brand": body.brand,
            "flavor": body.flavor,
            "category": body.category,
        },
    )

    for frame_path in draft.get("frame_paths", [])[:3]:
        try:
            img_vec = embed_model.encode_image(frame_path)
            angle = Path(frame_path).stem
            visual_id = f"{body.sku}_{angle}"
            vector_store.upsert_visual(
                visual_id,
                img_vec,
                {
                    "sku": body.sku,
                    "brand": body.brand,
                    "category": body.category,
                    "angle": angle,
                },
            )
            url = image_store.save_image(body.sku, frame_path, label=angle)
            product_db.add_image(body.sku, url, angle)
        except Exception as exc:
            _logger.warning("failed to index frame %s: %s", frame_path, exc)

    db.execute("UPDATE draft_jobs SET status='committed' WHERE job_id=?", (body.job_id,))
    db.commit()
    return CommitResponse(sku=body.sku)


@router.patch("/product/{sku}", response_model=CommitResponse)
async def patch_product(sku: str, body: ProductPatch, request: Request):
    product_db = request.app.state.product_db
    row = product_db.get_product(sku)
    if row is None:
        raise HTTPException(status_code=404, detail="product not found")

    updates = {k: v for k, v in body.model_dump(exclude_none=True).items() if k != "aliases"}
    if updates:
        product_db.update_product(sku, updates)

    for alias in body.aliases:
        product_db.add_alias(sku, alias)

    return CommitResponse(sku=sku)
