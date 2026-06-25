from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

MAX_DIM = 800
JPEG_QUALITY = 85


class ImageStore:
    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def save_image(self, product_id: str, src_path: str | Path, label: str = "main") -> str:
        product_dir = self._base / product_id
        product_dir.mkdir(parents=True, exist_ok=True)
        out_path = product_dir / f"{label}.jpg"
        img = Image.open(src_path).convert("RGB")
        img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
        img.save(str(out_path), "JPEG", quality=JPEG_QUALITY)
        return f"/images/{product_id}/{label}.jpg"

    def get_image_url(self, product_id: str, label: str = "main") -> Optional[str]:
        path = self._base / product_id / f"{label}.jpg"
        if not path.exists():
            return None
        return f"/images/{product_id}/{label}.jpg"

    def list_images(self, product_id: str) -> list[str]:
        d = self._base / product_id
        if not d.exists():
            return []
        return sorted(f"/images/{product_id}/{p.stem}.jpg" for p in d.glob("*.jpg"))
