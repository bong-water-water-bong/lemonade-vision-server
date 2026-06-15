# lemonade-vision-server — Wiki

> FastAPI server that identifies vape-shop products from iPhone multi-modal capture (video, stills, LiDAR depth, audio) and serves the result to lemonade-cashier's `sensors.*` layer.

## Runtime Setup

- This repo intentionally owns the heavier vision stack: FastAPI, Torch,
  CLIP, ChromaDB, VLM, and ASR integrations.
- Department repos do not install this service during their base
  `make install`.
- Run the service only when local product identification is needed; if
  VLM/ASR are unavailable, cashier must fall back to manual entry or
  lower-confidence verification.
- Product drafts are proposals. Cashier remains authoritative for SKU,
  price, sale close, and audit state.

## Current State

**What is working (by inspection):**
- All pipeline modules are implemented: frame extraction, barcode, VLM extraction, dimensions, background removal, CLIP embeddings
- Session lifecycle (create / validate / expire / close) is complete
- Capture upload endpoints (video, still, depth, audio, finalize) are implemented
- `DraftAssembler` orchestration is complete; finalize runs pipeline as background asyncio task
- SQLite schema with products, aliases, images, sessions, and draft_jobs tables
- ChromaDB dual-collection vector store (visual + text)
- `/deduce/text` and `/deduce/audio` endpoints with VLM signal enrichment and alias bonuses
- Confidence scorer with weighted aggregation and auto/verify/reject thresholds

**What is stubbed or incomplete:**
- `pipeline/background.py` (`remove_background`) is implemented but not wired into `DraftAssembler.run()` — embeddings and VLM see raw frames with backgrounds
- Embedding score in `assemble_draft()` is a fixed `0.5` placeholder, not a real ChromaDB similarity lookup
- `api/product.py` (commit, PATCH, GET) not read — contents unknown

**Test status:** All 12 test files fail at collection with `ModuleNotFoundError: No module named 'lemonade_vision'`. The package is not installed in the current Python environment. Run `uv pip install -e .` before any test work.

## Start Here

- [[architecture]] — read first: pipeline structure, store layout, session model, HTTP API
- [[scan-pipeline]] — if working on any pipeline stage
- [[confidence-scoring]] — if touching thresholds or the auto/verify/reject logic

## Open Threads

1. **Tests: package not installed** — `python3 -m pytest` fails across all 12 test files (`ModuleNotFoundError`). Install with `uv pip install -e .` and re-run to discover the actual failing tests vs passing tests.
2. **Embedding score placeholder** — `assemble_draft()` hardcodes `embedding_score = 0.5 if frame_paths else 0.0` rather than querying ChromaDB. Real similarity scoring against known products would make the confidence more meaningful for repeat scans.
3. **background.py not wired** — `remove_background()` exists and is presumably tested, but `DraftAssembler.run()` never calls it. Wiring it in before VLM/embedding submission would improve visual signal quality in busy capture environments.
4. **fw-server ASR URL not configurable** — Both `DraftAssembler` and `api/deduce.py` hardcode `http://localhost:8004`. There is no `VISION_ASR_URL` env var override in `create_app()`, unlike the data dir which reads `VISION_DATA_DIR`.

## Article Index

| Article | What it covers |
|---------|----------------|
| [[architecture]] | System shape, pipeline → store → session flow, HTTP API |
| [[scan-pipeline]] | Each pipeline stage: barcode, VLM, CLIP embeddings, dimensions, frames, background |
| [[confidence-scoring]] | Score aggregation, thresholds (0.85/0.50), auto/verify/reject decision |
