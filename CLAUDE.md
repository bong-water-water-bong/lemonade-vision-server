> **Start here:** Read `docs/wiki/README.md` before any work on this project.

# CLAUDE.md — conventions for lemonade-vision-server

## Hard rules

- **Install before testing.** The package is not installed by default. Run `uv pip install -e .` from the project root before running `pytest`. All tests will fail with `ModuleNotFoundError` otherwise.

- **Do not amend `assemble_draft`'s embedding score without a ChromaDB lookup.** The `embedding_score = 0.5 if frame_paths else 0.0` line in `draft.py` is a deliberate placeholder (0 = no frames, 0.5 = frames present but not yet compared against known products). Do not replace it with a constant other than 0.5 without wiring in a real `VectorStore.query_visual()` call and passing the best-match distance through to `compute_confidence`.

- **Exception suppression in pipeline stages is intentional.** Every stage in `DraftAssembler.run()` wraps its call in `try/except Exception` so a single failing stage does not abort the whole scan. Do not remove these guards or let exceptions propagate to `_run_pipeline` — the only safe propagation path is updating `draft_jobs.status = 'failed'` via the outer `except` in `_run_pipeline`.

- **Session tmp_dirs are always under `sessions_path`.** `create_session` receives a `tmp_dir` created by `tempfile.mkdtemp(dir=str(sessions_path))`. Never write session data outside this subtree; `close_session` and `expire_old_sessions` clean up by calling `shutil.rmtree` on `row["tmp_dir"]`.

- **VLM base URL is `http://localhost:8001`, ASR is `http://localhost:8004`.** These are the Lemonade NPU inference server and the faster-whisper fw-server respectively. The ASR URL is currently hardcoded in two places (`DraftAssembler.__init__` and `api/deduce.py`); until a `VISION_ASR_URL` env var is added, both must be kept in sync.
