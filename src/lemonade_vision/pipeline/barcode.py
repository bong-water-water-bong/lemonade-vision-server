from __future__ import annotations
from pathlib import Path
from typing import Optional
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode


def extract_upc(image_path: Path) -> Optional[str]:
    try:
        img = Image.open(image_path)
    except Exception:
        return None
    results = pyzbar_decode(img)
    for r in results:
        if r.type in ("EAN13", "UPCA", "UPCE", "EAN8", "CODE128", "CODE39"):
            return r.data.decode("utf-8")
    return None
