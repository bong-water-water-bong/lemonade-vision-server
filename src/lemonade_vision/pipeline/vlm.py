# src/lemonade_vision/pipeline/vlm.py
from __future__ import annotations
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Optional
import httpx


ONBOARD_TIMEOUT = 15.0
DEDUCE_TIMEOUT = 3.0

EXTRACT_PROMPT = """\
You are a product identification assistant for a vape shop inventory system.
Analyse the provided product images and narration transcript (if any).
Return ONLY a valid JSON object with these exact keys:
{
  "brand": string or null,
  "flavor": string or null,
  "category": string or null,
  "puff_count": integer or null,
  "nicotine_mg": integer or null,
  "ocr_text": string or null,
  "warnings": [string],
  "confidence": float 0-1
}
Typical categories: disposable_vape, e_liquid, pod, device, accessory.
"""

DEDUCE_PROMPT = """\
Extract structured product signals from this customer query for a vape shop.
Return ONLY a valid JSON object:
{
  "brand": string or null,
  "flavor": string or null,
  "size": string or null,
  "color": string or null,
  "category": string or null
}
Query: {query}
"""


@dataclass
class VLMResult:
    brand: Optional[str] = None
    flavor: Optional[str] = None
    category: Optional[str] = None
    puff_count: Optional[int] = None
    nicotine_mg: Optional[int] = None
    ocr_text: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    vlm_status: str = "ok"


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


class VLMClient:
    def __init__(self, base_url: str = "http://localhost:8001") -> None:
        self._base_url = base_url
        self._http = httpx.AsyncClient(base_url=base_url, timeout=ONBOARD_TIMEOUT)

    async def extract_product_info(
        self,
        image_paths: list[str],
        narration: Optional[str],
    ) -> VLMResult:
        content: list[dict] = [{"type": "text", "text": EXTRACT_PROMPT}]
        for path in image_paths[:4]:
            try:
                b64 = _encode_image(path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
            except Exception:
                pass
        if narration:
            content.append({"type": "text", "text": f"Operator narration: {narration}"})

        try:
            resp = await self._http.post(
                "/v1/chat/completions",
                json={
                    "model": "local",
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.1,
                },
                timeout=ONBOARD_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(_strip_fences(raw))
        except Exception:
            return VLMResult(vlm_status="unavailable")

        return VLMResult(
            brand=data.get("brand"),
            flavor=data.get("flavor"),
            category=data.get("category"),
            puff_count=data.get("puff_count"),
            nicotine_mg=data.get("nicotine_mg"),
            ocr_text=data.get("ocr_text"),
            warnings=data.get("warnings", []),
            confidence=float(data.get("confidence", 0.0)),
            vlm_status="ok",
        )

    async def deduce_product_signals(self, query: str) -> dict:
        content = DEDUCE_PROMPT.replace("{query}", query)
        try:
            resp = await self._http.post(
                "/v1/chat/completions",
                json={
                    "model": "local",
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.0,
                },
                timeout=DEDUCE_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return json.loads(_strip_fences(raw))
        except Exception:
            return {}
