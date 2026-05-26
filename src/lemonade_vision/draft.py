# src/lemonade_vision/draft.py
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

from lemonade_vision.pipeline.background import remove_background
from lemonade_vision.pipeline.barcode import extract_upc
from lemonade_vision.pipeline.dimensions import depth_to_dimensions
from lemonade_vision.pipeline.frames import frames_from_video
from lemonade_vision.pipeline.vlm import VLMResult

_logger = logging.getLogger(__name__)


def assemble_draft(
    job_id: str,
    session_id: str,
    vlm_result: VLMResult,
    upc: Optional[str],
    dimensions: Optional[tuple[float, float, float]],
    narration: Optional[str],
    frame_paths: list[str],
) -> dict:
    upc_score = 1.0 if upc else 0.0
    vlm_score = vlm_result.confidence if vlm_result.vlm_status == "ok" else 0.0
    dim_score = 0.5 if dimensions else 0.0
    embedding_score = 0.5 if frame_paths else 0.0

    signal_scores = {
        "upc": upc_score,
        "vlm": vlm_score,
        "embedding": embedding_score,
        "dimension": dim_score,
    }

    dim_data = None
    if dimensions:
        w, h, d = dimensions
        dim_data = {"width_mm": w, "height_mm": h, "depth_mm": d}

    return {
        "job_id": job_id,
        "session_id": session_id,
        "status": "ready",
        "upc": upc,
        "brand": vlm_result.brand,
        "flavor": vlm_result.flavor,
        "category": vlm_result.category,
        "puff_count": vlm_result.puff_count,
        "nicotine_mg": vlm_result.nicotine_mg,
        "ocr_text": vlm_result.ocr_text,
        "narration": narration,
        "dimensions": dim_data,
        "signal_scores": signal_scores,
        "vlm_status": vlm_result.vlm_status,
        "frame_paths": frame_paths,
    }


class DraftAssembler:
    """Orchestrates the full onboarding pipeline from session data to draft record."""

    def __init__(self, vlm_client, embedding_model, fw_base_url: str = "http://localhost:8004"):
        self._vlm = vlm_client
        self._embed = embedding_model
        self._fw_base_url = fw_base_url

    async def run(
        self,
        job_id: str,
        session_id: str,
        rotation_video_path: Optional[str],
        still_paths: dict[str, str],
        depth_path: Optional[str],
        narration_path: Optional[str],
        frame_out_dir: str,
    ) -> dict:
        # 1. Extract frames from rotation video
        frame_paths: list[str] = []
        if rotation_video_path:
            try:
                frame_paths = frames_from_video(
                    Path(rotation_video_path), Path(frame_out_dir)
                )
            except Exception as exc:
                _logger.warning("frames_from_video failed: %s", exc)

        # Add close-up stills (deduplicated)
        seen_paths: set[str] = set(frame_paths)
        for path in still_paths.values():
            if path not in seen_paths:
                frame_paths.append(path)
                seen_paths.add(path)

        # 1.5 Background removal — preprocess frames for VLM extraction
        bg_frame_paths: list[str] = list(frame_paths)
        try:
            bg_out_dir = Path(frame_out_dir) / "bg"
            bg_out_dir.mkdir(parents=True, exist_ok=True)
            bg_frame_paths = []
            for fp in frame_paths:
                in_path = Path(fp)
                out_path = bg_out_dir / f"bg_{in_path.name}"
                bg_frame_paths.append(str(await asyncio.to_thread(remove_background, in_path, out_path)))
        except Exception as exc:
            _logger.warning("background_removal stage failed: %s", exc)
            bg_frame_paths = list(frame_paths)

        # 2. Barcode from UPC still (preferred) or any frame
        upc: Optional[str] = None
        if "upc" in still_paths:
            upc = extract_upc(Path(still_paths["upc"]))
        if upc is None:
            for fp in frame_paths[:5]:
                upc = extract_upc(Path(fp))
                if upc:
                    break

        # 3. Transcribe narration
        narration: Optional[str] = None
        if narration_path:
            narration = await self._transcribe(narration_path)

        # 4. VLM extraction
        try:
            vlm_result = await self._vlm.extract_product_info(
                bg_frame_paths[:4], narration=narration
            )
        except Exception as exc:
            _logger.warning("vlm extraction stage failed: %s", exc)
            vlm_result = VLMResult(vlm_status="unavailable")

        # 5. Dimensions from depth
        dimensions: Optional[tuple[float, float, float]] = None
        if depth_path:
            try:
                grid = np.load(depth_path) if depth_path.endswith(".npy") else \
                       np.array(json.loads(Path(depth_path).read_text()))
                dimensions = depth_to_dimensions(grid)
            except Exception as exc:
                _logger.warning("depth_to_dimensions failed: %s", exc)

        return assemble_draft(
            job_id=job_id,
            session_id=session_id,
            vlm_result=vlm_result,
            upc=upc,
            dimensions=dimensions,
            narration=narration,
            frame_paths=frame_paths,
        )

    async def _transcribe(self, audio_path: str) -> Optional[str]:
        try:
            with open(audio_path, "rb") as f:
                data = f.read()
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._fw_base_url}/transcribe",
                    content=data,
                    headers={"Content-Type": "audio/wav"},
                )
                resp.raise_for_status()
                return resp.json().get("text")
        except Exception:
            return None
