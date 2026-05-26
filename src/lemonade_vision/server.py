# src/lemonade_vision/server.py
from __future__ import annotations
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from lemonade_vision.models import HealthResponse
from lemonade_vision.store.schema import init_db
from lemonade_vision.store.product_db import ProductDB
from lemonade_vision.store.vector_db import VectorStore
from lemonade_vision.store.image_store import ImageStore
from lemonade_vision.pipeline.vlm import VLMClient
from lemonade_vision.pipeline.embeddings import EmbeddingModel
from lemonade_vision.draft import DraftAssembler
from lemonade_vision.session import create_session
from lemonade_vision.api.capture import router as capture_router
from lemonade_vision.api.product import router as product_router
from lemonade_vision.api.deduce import router as deduce_router


def create_app(data_dir: str | None = None) -> FastAPI:
    if data_dir is None:
        data_dir = os.environ.get(
            "VISION_DATA_DIR",
            str(Path.home() / "lemonade-vision-server" / "data")
        )

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    db_path = data_path / "products.db"
    chroma_path = data_path / "chroma"
    images_path = data_path / "images"
    sessions_path = data_path / "sessions"
    sessions_path.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = init_db(db_path)
        product_db = ProductDB(db)
        vector_store = VectorStore(str(chroma_path))
        image_store = ImageStore(str(images_path))
        vlm_client = VLMClient(base_url="http://localhost:8001")
        embed_model = EmbeddingModel()
        assembler = DraftAssembler(vlm_client, embed_model, vector_store=vector_store)

        app.state.db = db
        app.state.product_db = product_db
        app.state.vector_store = vector_store
        app.state.image_store = image_store
        app.state.vlm_client = vlm_client
        app.state.embed_model = embed_model
        app.state.assembler = assembler
        app.state.sessions_path = str(sessions_path)

        yield

        db.close()

    images_path.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="lemonade-vision-server", version="0.1.0", lifespan=lifespan)

    app.include_router(capture_router)
    app.include_router(product_router)
    app.include_router(deduce_router)
    app.mount("/images", StaticFiles(directory=str(images_path)), name="images")

    @app.post("/session/start")
    async def session_start(request: Request):
        import qrcode
        import io
        import base64
        tmp_dir = tempfile.mkdtemp(dir=str(sessions_path))
        sid = create_session(request.app.state.db, tmp_dir)
        qr = qrcode.make(sid)
        buf = io.BytesIO()
        qr.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"session_id": sid, "qr_png_b64": b64}

    @app.delete("/session/{session_id}", status_code=204)
    async def session_delete(session_id: str, request: Request):
        from lemonade_vision.session import close_session
        close_session(request.app.state.db, session_id)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        import httpx
        vlm_ok = False
        try:
            resp = httpx.get("http://localhost:8001/v1/models", timeout=2.0)
            vlm_ok = resp.status_code == 200
        except Exception:
            pass
        count = request.app.state.vector_store.product_count()
        return HealthResponse(
            status="ok", vlm_reachable=vlm_ok, chroma_product_count=count
        )

    @app.get("/pairing/qr")
    async def pairing_qr(session_id: str, request: Request):
        import qrcode
        import io
        import base64
        qr = qrcode.make(session_id)
        buf = io.BytesIO()
        qr.save(buf, format="PNG")
        return {"qr_png_b64": base64.b64encode(buf.getvalue()).decode()}

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8787)
