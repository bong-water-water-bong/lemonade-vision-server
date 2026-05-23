# src/lemonade_vision/api/capture.py
from __future__ import annotations
import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from lemonade_vision.models import FinalizeResponse
from lemonade_vision.session import validate_session

router = APIRouter()

VALID_ANGLES = {"upc", "label", "front", "rear", "left", "right", "top", "bottom"}


def _get_session_token(request: Request) -> str:
    token = request.headers.get("X-Session-Token")
    if not token:
        raise HTTPException(status_code=401, detail="X-Session-Token header required")
    return token


def _require_session(token: Annotated[str, Depends(_get_session_token)], request: Request):
    db = request.app.state.db
    row = validate_session(db, token)
    if row is None:
        raise HTTPException(status_code=401, detail="Session expired or not found")
    return dict(row)


@router.post("/capture/video", status_code=202)
async def upload_video(
    request: Request,
    session: Annotated[dict, Depends(_require_session)],
    file: UploadFile = File(...),
):
    tmp_dir = Path(session["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_dir / "rotation.mp4"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    db = request.app.state.db
    db.execute(
        "UPDATE capture_sessions SET frame_count = 0 WHERE session_id = ?",
        (session["session_id"],),
    )
    db.commit()
    return {"message": "video received", "path": str(video_path)}


@router.post("/capture/still")
async def upload_still(
    request: Request,
    session: Annotated[dict, Depends(_require_session)],
    angle: str = Form(...),
    file: UploadFile = File(...),
):
    if angle not in VALID_ANGLES:
        raise HTTPException(
            status_code=422,
            detail=f"angle must be one of {sorted(VALID_ANGLES)}",
        )
    tmp_dir = Path(session["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "still.jpg").suffix or ".jpg"
    frame_id = str(uuid.uuid4())[:8]
    out_path = tmp_dir / f"still_{angle}_{frame_id}{suffix}"
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"frame_id": frame_id, "angle": angle, "path": str(out_path)}


@router.post("/capture/depth")
async def upload_depth(
    request: Request,
    session: Annotated[dict, Depends(_require_session)],
    frame_id: str = Form(...),
    file: UploadFile = File(...),
):
    tmp_dir = Path(session["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"depth_{frame_id}.json"
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": "depth received", "frame_id": frame_id}


@router.post("/capture/audio")
async def upload_audio(
    request: Request,
    session: Annotated[dict, Depends(_require_session)],
    file: UploadFile = File(...),
):
    tmp_dir = Path(session["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "narration.wav").suffix or ".wav"
    narration_path = tmp_dir / f"narration{suffix}"
    with open(narration_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    db = request.app.state.db
    db.execute(
        "UPDATE capture_sessions SET narration_path = ? WHERE session_id = ?",
        (str(narration_path), session["session_id"]),
    )
    db.commit()
    return {"message": "audio received"}


@router.post("/capture/finalize", response_model=FinalizeResponse)
async def finalize(
    request: Request,
    session: Annotated[dict, Depends(_require_session)],
):
    job_id = str(uuid.uuid4())
    db = request.app.state.db
    from datetime import datetime, timezone
    db.execute(
        "INSERT INTO draft_jobs (job_id, session_id, status, created_at) VALUES (?,?,?,?)",
        (job_id, session["session_id"], "processing",
         datetime.now(timezone.utc).isoformat()),
    )
    db.commit()

    assembler = request.app.state.assembler
    tmp_dir = Path(session["tmp_dir"])
    asyncio.create_task(
        _run_pipeline(db, assembler, job_id, session, tmp_dir, request.app.state)
    )
    return FinalizeResponse(job_id=job_id)


async def _run_pipeline(db, assembler, job_id, session, tmp_dir, state):
    import json as _json
    import traceback
    try:
        video_path = tmp_dir / "rotation.mp4"
        narration_path = session.get("narration_path")

        still_paths: dict[str, str] = {}
        for p in tmp_dir.glob("still_*.jpg"):
            parts = p.stem.split("_")
            if len(parts) >= 2:
                still_paths[parts[1]] = str(p)

        depth_candidates = list(tmp_dir.glob("depth_*.json"))
        depth_path = str(depth_candidates[0]) if depth_candidates else None

        frame_out_dir = tmp_dir / "frames"
        draft = await assembler.run(
            job_id=job_id,
            session_id=session["session_id"],
            rotation_video_path=str(video_path) if video_path.exists() else None,
            still_paths=still_paths,
            depth_path=depth_path,
            narration_path=narration_path,
            frame_out_dir=str(frame_out_dir),
        )

        signal_scores = draft.pop("signal_scores", {})
        db.execute(
            "UPDATE draft_jobs SET status='ready', draft_json=?, signal_scores=? WHERE job_id=?",
            (_json.dumps(draft), _json.dumps(signal_scores), job_id),
        )
        db.commit()
    except Exception as exc:
        db.execute(
            "UPDATE draft_jobs SET status='failed', draft_json=? WHERE job_id=?",
            (_json.dumps({"error": str(exc), "traceback": traceback.format_exc()}), job_id),
        )
        db.commit()
