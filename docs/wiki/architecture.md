# Architecture

> lemonade-vision-server is a FastAPI service that identifies vape-shop products from multi-modal capture data (video, stills, depth, audio) and exposes the result to lemonade-cashier's `sensors.*` layer.

## Overview

The cashier's `sensors.*` stubs need a real product identification backend capable of more than a UPC lookup. This server fills that gap by running four parallel identification signals — barcode scan, VLM visual analysis, CLIP embedding similarity, and LiDAR-derived physical dimensions — then aggregating them into a confidence score that drives an auto/verify/reject decision.

The service is designed for an iPhone 15 Pro Max as the capture device: the client uploads a 360° rotation video, angle-labeled stills, a LiDAR depth JSON, and optional operator voice narration, then calls `/capture/finalize`. The server processes everything asynchronously and stores a draft record the cashier can poll or commit.

## How It Works

```
iPhone client
   │
   ├─ POST /capture/video       (rotation .mp4)
   ├─ POST /capture/still       (angle-labeled JPEG per face)
   ├─ POST /capture/depth       (ARKit depth grid as JSON)
   ├─ POST /capture/audio       (operator narration WAV)
   └─ POST /capture/finalize ──► asyncio.create_task(_run_pipeline)
                                        │
                              DraftAssembler.run()
                                        │
                          ┌─────────────┴──────────────┐
                          │  pipeline stages (serial)  │
                          │                            │
                          │  1. frames_from_video      │ ← ffmpeg + blur filter
                          │  2. extract_upc            │ ← pyzbar on stills/frames
                          │  3. _transcribe (fw-server)│ ← :8004 ASR
                          │  4. VLMClient.extract_*    │ ← :8001 VLM
                          │  5. depth_to_dimensions    │ ← numpy ARKit math
                          │                            │
                          └─────────────┬──────────────┘
                                        │
                              assemble_draft()
                              compute_confidence()
                                        │
                          ┌─────────────┴──────────────┐
                          │  SQLite draft_jobs         │ status: processing → ready
                          └─────────────┬──────────────┘
                                        │
                          GET /draft/<job_id>   ← cashier polls
                          POST /product/commit  ← cashier or human commits
                                        │
                          ┌─────────────┴──────────────┐
                          │  SQLite products           │ permanent product record
                          │  ChromaDB visual+text      │ CLIP embeddings for search
                          └────────────────────────────┘
```

**`pipeline/`** — Six single-responsibility modules called in sequence by `DraftAssembler.run()`. Each module is a thin function or client that either returns a result or swallows its own exceptions and returns a safe default (None / VLMResult(vlm_status="unavailable")). Nothing in the pipeline raises to the assembler.

**`store/`** — Three storage backends initialized at startup and injected into `app.state`: `ProductDB` (SQLite for structured product data), `VectorStore` (ChromaDB with two collections — `product_visual` for image embeddings, `product_text` for text embeddings), and `ImageStore` (filesystem directory served as `/images` static mount).

**`session.py`** — Captures are grouped by `capture_session`. A session is created via `POST /session/start` (returns a QR pairing code for the iPhone client). A session is a UUID tied to a temporary directory and a 10-minute TTL stored in SQLite `capture_sessions`. The client authenticates every capture upload via `X-Session-Token` header. Sessions are cleaned up on expiry or explicit `DELETE /session/{session_id}`, which removes both the DB row and the `tmp_dir` subtree.

**`server.py`** — `create_app()` factory wires together all state at startup via FastAPI `lifespan`. The VLM client points to `http://localhost:8001` (the Lemonade/FLM NPU inference server). The ASR client points to `http://localhost:8004` (fw-server). The `/health` endpoint probes the VLM and reports ChromaDB product count, making it suitable for cashier's readiness check.

**`api/`** — Three routers: `capture` (session-gated upload endpoints + finalize), `product` (commit draft → product record, PATCH, GET), `deduce` (text or audio query → ranked candidates from vector store). The deduce flow runs VLM signal extraction to enrich the query before embedding, then adds brand/flavor/alias bonuses on top of cosine similarity.

## Key Decisions

- **Multi-signal rather than UPC-only**: Vape disposables frequently have missing, damaged, or non-standard barcodes. UPC carries 40% weight; the other three signals (VLM 30%, embedding 20%, dimension 10%) provide fallback identification when UPC is absent, making the system usable for unlabeled or pre-market products.

- **SQLite for structured data, ChromaDB for vectors**: SQLite is zero-infrastructure, transactional, and ships as part of Python — appropriate for a single-node kiosk deployment. ChromaDB is used only for the nearest-neighbor vector queries where SQL cannot help; it persists to a directory alongside the SQLite file.

- **Per-scan sessions with TTL and tmp_dir cleanup**: Each capture session creates an isolated temporary directory. The 10-minute TTL and `shutil.rmtree` on close prevent uncleaned frame/depth/audio files from accumulating. Session expiry is lazy (checked on access) rather than scheduled, so there is no background sweeper thread.

- **`asyncio.create_task` for pipeline execution**: `/capture/finalize` returns HTTP 200 (FastAPI default — no `status_code` argument on the route decorator) and runs the pipeline as a background asyncio task. This prevents the iPhone client from hanging during a 10–15 second VLM call. The `/capture/video` endpoint returns 202. The finalize status diverges from async design intent and could be corrected by adding `status_code=202` to the `/capture/finalize` route decorator. The cashier polls `draft_jobs.status` until it transitions from `processing` → `ready` or `failed`.

- **VLM at localhost:8001 with hard timeout**: The VLM client uses a 15-second `ONBOARD_TIMEOUT` for product extraction and a 3-second `DEDUCE_TIMEOUT` for query signal extraction. If the VLM is unreachable, `VLMResult(vlm_status="unavailable")` is returned and the VLM score contribution drops to 0.0 — the scan continues with reduced confidence rather than failing.

## Gotchas

- **Package not installed — all tests fail at collection**: The test suite imports `lemonade_vision` by package name, but `python3 -m pytest` is run from the project root without the package being installed. You must run `pip install -e .` (or `uv pip install -e .`) in the virtualenv before tests will collect. All 12 test files fail with `ModuleNotFoundError: No module named 'lemonade_vision'` and `No module named 'fastapi'`.

- **Depth path handling is fragile**: `DraftAssembler.run()` expects depth data as either `.npy` (numpy binary) or JSON, but `api/capture.py` always saves the upload as `depth_<frame_id>.json`. If the client sends a numpy binary with a `.json` extension the `json.loads` branch will fail silently (caught by `except Exception`) and `dimensions` will be None. The score for the `dimension` signal drops to 0.0 with no error surfaced to the caller.

- **`assemble_draft` uses placeholder embedding score**: The `embedding_score` in `assemble_draft()` is set to `0.5 if frame_paths else 0.0` — it does not actually run any embedding similarity search against ChromaDB. Real embedding similarity is only computed in the `/deduce/text` route. The `0.5` constant means scans with any frames always get a fixed embedding contribution, which inflates confidence for new products not yet in the database.

- **Session expiry is string-compared ISO timestamps**: `validate_session` compares `row["expires_at"] < _now_iso()` as a plain string comparison. This works correctly only because both sides are UTC ISO 8601 strings in the same format. Any timezone offset or format inconsistency would silently allow expired sessions.

- **fw-server hardcoded to port 8004**: Both `DraftAssembler` (narration transcription) and `api/deduce.py` (audio deduce) hardcode `http://localhost:8004` for the faster-whisper ASR. There is no environment variable override for the ASR URL in the server factory.

## Related

- [[scan-pipeline]] — detailed breakdown of each pipeline stage
- [[confidence-scoring]] — how the four signal scores combine into auto/verify/reject
