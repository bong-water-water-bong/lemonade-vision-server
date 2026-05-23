from __future__ import annotations
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

_UPDATABLE_FIELDS = frozenset({
    "barcode", "brand", "flavor", "category",
    "puff_count", "nicotine_mg", "confidence",
    "requires_attendant", "width_mm", "height_mm", "depth_mm",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProductDB:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._db = conn

    def insert_product(self, data: dict) -> str:
        product_id = str(uuid.uuid4())
        now = _now()
        self._db.execute(
            """INSERT INTO products
               (product_id, barcode, brand, flavor, category,
                puff_count, nicotine_mg, confidence, requires_attendant,
                width_mm, height_mm, depth_mm, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                product_id,
                data.get("barcode"),
                data.get("brand", ""),
                data.get("flavor", ""),
                data.get("category", ""),
                data.get("puff_count"),
                data.get("nicotine_mg"),
                data.get("confidence", 0.85),
                int(data.get("requires_attendant", False)),
                data.get("width_mm"),
                data.get("height_mm"),
                data.get("depth_mm"),
                now,
                now,
            ),
        )
        self._db.commit()
        return product_id

    def get_product(self, product_id: str) -> Optional[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()

    def add_alias(self, product_id: str, alias: str) -> None:
        self._db.execute(
            "INSERT INTO product_aliases (product_id, alias) VALUES (?, ?)",
            (product_id, alias),
        )
        self._db.commit()

    def get_aliases(self, product_id: str) -> list[str]:
        rows = self._db.execute(
            "SELECT alias FROM product_aliases WHERE product_id = ?", (product_id,)
        ).fetchall()
        return [r["alias"] for r in rows]

    def add_image(self, product_id: str, url: str, label: str) -> None:
        self._db.execute(
            "INSERT INTO product_images (product_id, url, label) VALUES (?,?,?)",
            (product_id, url, label),
        )
        self._db.commit()

    def update_product(self, product_id: str, fields: dict) -> None:
        if not fields:
            return
        invalid = set(fields) - _UPDATABLE_FIELDS
        if invalid:
            raise ValueError(f"Non-updatable field(s): {invalid}")
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [_now(), product_id]
        self._db.execute(
            f"UPDATE products SET {set_clause}, updated_at = ? WHERE product_id = ?",
            values,
        )
        self._db.commit()
