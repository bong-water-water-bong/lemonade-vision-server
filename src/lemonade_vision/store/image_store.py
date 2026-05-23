from __future__ import annotations
from pathlib import Path
from PIL import Image

MAX_DIM = 800
JPEG_QUALITY = 85


class ImageStore:
    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def save_image(self, sku: str, angle: str, src: Path) -> Path:
        sku_dir = self._base / sku
        sku_dir.mkdir(parents=True, exist_ok=True)
        out_path = sku_dir / f"{angle}.jpg"
        img = Image.open(src).convert("RGB")
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        img.save(str(out_path), "JPEG", quality=JPEG_QUALITY)
        return out_path

    def get_image_url(self, sku: str, angle: str) -> str:
        return f"/images/{sku}/{angle}.jpg"

    def list_images(self, sku: str) -> list[Path]:
        d = self._base / sku
        if not d.exists():
            return []
        return sorted(d.glob("*.jpg"))
