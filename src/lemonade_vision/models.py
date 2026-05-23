from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class SessionStartResponse(BaseModel):
    session_id: str
    qr_png_b64: str


class FinalizeResponse(BaseModel):
    job_id: str
    message: str = "processing"


class SignalScores(BaseModel):
    upc: float = 0.0
    vlm: float = 0.0
    embedding: float = 0.0
    dimension: float = 0.0


class ProductDimensions(BaseModel):
    width_mm: float
    height_mm: float
    depth_mm: float


class DraftProduct(BaseModel):
    job_id: str
    status: str
    upc: Optional[str] = None
    brand: Optional[str] = None
    flavor: Optional[str] = None
    category: Optional[str] = None
    puff_count: Optional[int] = None
    nicotine_mg: Optional[int] = None
    ocr_text: Optional[str] = None
    narration: Optional[str] = None
    dimensions: Optional[ProductDimensions] = None
    signal_scores: Optional[SignalScores] = None
    vlm_status: str = "ok"
    reference_image_urls: list[str] = []


class CommitRequest(BaseModel):
    job_id: str
    sku: str
    brand: str
    flavor: str
    category: str
    puff_count: Optional[int] = None
    nicotine_mg: Optional[int] = None
    requires_attendant: bool = False
    confidence_threshold: float = 0.85
    aliases: list[str] = []


class CommitResponse(BaseModel):
    sku: str
    message: str = "committed"


class ProductPatch(BaseModel):
    brand: Optional[str] = None
    flavor: Optional[str] = None
    category: Optional[str] = None
    puff_count: Optional[int] = None
    nicotine_mg: Optional[int] = None
    requires_attendant: Optional[bool] = None
    confidence_threshold: Optional[float] = None
    aliases: list[str] = []


class DeduceRequest(BaseModel):
    query: str
    top_k: int = 3


class DeduceCandidate(BaseModel):
    sku: str
    confidence: float
    match_reason: str
    brand: Optional[str] = None
    flavor: Optional[str] = None


class DeduceResponse(BaseModel):
    candidates: list[DeduceCandidate]
    query_used: str = ""


class HealthResponse(BaseModel):
    status: str
    vlm_reachable: bool
    chroma_product_count: int
