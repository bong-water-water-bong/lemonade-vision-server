from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProductDB:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._db = conn

    def insert_product(
        self,
        sku: str,
        brand: str,
        flavor: str,
        category: str,
        upc: Optional[str] = None,
        puff_count: Optional[int] = None,
        nicotine_mg: Optional[int] = None,
        ocr_text: Optional[str] = None,
        narration: Optional[str] = None,
        width_mm: Optional[float] = None,
        height_mm: Optional[float] = None,
        depth_mm: Optional[float] = None,
        confidence_threshold: float = 0.85,
        requires_attendant: bool = False,
    ) -> None:
        now = _now()
        self._db.execute(
            """INSERT INTO products
               (sku,upc,brand,flavor,category,puff_count,nicotine_mg,
                ocr_text,narration,width_mm,height_mm,depth_mm,
                confidence_threshold,requires_attendant,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sku, upc, brand, flavor, category, puff_count, nicotine_mg,
             ocr_text, narration, width_mm, height_mm, depth_mm,
             confidence_threshold, int(requires_attendant), now, now),
        )
        self._db.commit()

    def get_product(self, sku: str) -> Optional[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM products WHERE sku = ?", (sku,)
        ).fetchone()

    def add_alias(self, sku: str, alias: str) -> None:
        self._db.execute(
            "INSERT INTO product_aliases (sku, alias) VALUES (?, ?)", (sku, alias)
        )
        self._db.commit()

    def get_aliases(self, sku: str) -> list[str]:
        rows = self._db.execute(
            "SELECT alias FROM product_aliases WHERE sku = ?", (sku,)
        ).fetchall()
        return [r["alias"] for r in rows]

    def add_image(self, sku: str, angle: str, path: str, is_primary: bool = False) -> None:
        self._db.execute(
            "INSERT INTO product_images (sku, angle, path, is_primary) VALUES (?,?,?,?)",
            (sku, angle, path, int(is_primary)),
        )
        self._db.commit()

    def update_product(self, sku: str, **kwargs) -> None:
        if not kwargs:
            return
        fields = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [_now(), sku]
        self._db.execute(
            f"UPDATE products SET {fields}, updated_at = ? WHERE sku = ?", values
        )
        self._db.commit()
