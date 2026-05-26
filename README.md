# lemonade-vision-server

[![ci](https://github.com/bong-water-water-bong/lemonade-vision-server/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bong-water-water-bong/lemonade-vision-server/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![local-first](https://img.shields.io/badge/local--first-strix%20halo-2ea44f)](#hardware)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> FastAPI product-identification server for the Lemonade Store ecosystem — iPhone multi-modal capture (video, stills, LiDAR, audio) → confidence-scored product match → `sensors.*` layer of lemonade-cashier.

**→ [Project Wiki](docs/wiki/README.md)** — architecture, decisions, gotchas, and agent onboarding.

`lemonade-vision-server` is the vision pipeline component of the [Lemonade Store](https://github.com/bong-water-water-bong/lemonade-store) suite. It receives iPhone capture sessions, runs a six-stage pipeline (barcode, VLM, CLIP embeddings, dimensions, background removal), scores confidence across signals, and returns a structured product draft to lemonade-cashier's sensor fusion layer.

## What it does

```text
iPhone capture session
  ├── video frames / stills     → barcode → VLM (Qwen2-VL) → CLIP embeddings
  ├── LiDAR depth map           → dimension estimation
  └── audio clip (opt)          → ASR label extraction
                                ↓
                    confidence scorer (auto / verify / reject)
                                ↓
                    SQLite product draft + ChromaDB vector store
                                ↓
                    lemonade-cashier sensors.* layer
```

## Hardware

Designed for AMD Ryzen AI MAX+ 395 (Strix Halo) on Ubuntu 26.04. Runs on any x86_64 Linux with a GPU; Qwen2-VL VLM inference benefits from the Radeon 8060S iGPU via ROCm.

## Install

```bash
git clone https://github.com/bong-water-water-bong/lemonade-vision-server.git
cd lemonade-vision-server
uv pip install -e ".[dev]"
```

## Run

```bash
uvicorn lemonade_vision.main:create_app --factory --host 127.0.0.1 --port 8005
```

Health check: `GET http://127.0.0.1:8005/health`

## Test

```bash
# Install first — tests fail at collection without the package installed
uv pip install -e .
make test
```

## API at a glance

| Endpoint | Method | Purpose |
|---|---|---|
| `/capture/session` | POST | Create a new capture session |
| `/capture/video` | POST | Upload video frames (202 async) |
| `/capture/still` | POST | Upload still image |
| `/capture/depth` | POST | Upload LiDAR depth map |
| `/capture/audio` | POST | Upload audio clip |
| `/capture/finalize` | POST | Run pipeline, return draft |
| `/deduce/text` | POST | Text-query product lookup |
| `/deduce/audio` | POST | Audio-query product lookup |
| `/product/commit` | POST | Commit draft to product catalog |

## Layout

```text
src/lemonade_vision/
  main.py            # FastAPI app factory
  api/               # Route handlers (capture, deduce, product)
  pipeline/          # Six-stage scan pipeline
  store/             # SQLite + ChromaDB persistence
  session/           # Capture session lifecycle
tests/               # pytest suites (12 files)
docs/wiki/           # Architecture, pipeline, confidence-scoring articles
```

## Status

v0.1. All pipeline stages implemented. Open threads:

- Embedding score is a placeholder `0.5` — real ChromaDB similarity lookup not yet wired
- ASR URL (`http://localhost:8004`) hardcoded — no `VISION_ASR_URL` env override

See [Project Wiki](docs/wiki/README.md) for full details.

## License

MIT.
