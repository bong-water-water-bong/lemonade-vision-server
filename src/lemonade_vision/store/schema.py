import sqlite3
from pathlib import Path


DDL = """
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    barcode TEXT,
    brand TEXT NOT NULL,
    flavor TEXT NOT NULL,
    category TEXT NOT NULL,
    puff_count INTEGER,
    nicotine_mg INTEGER,
    confidence REAL NOT NULL DEFAULT 0.85,
    requires_attendant BOOLEAN NOT NULL DEFAULT 0,
    width_mm REAL,
    height_mm REAL,
    depth_mm REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS product_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    alias TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS product_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT 'main'
);

CREATE TABLE IF NOT EXISTS capture_sessions (
    session_id TEXT PRIMARY KEY,
    tmp_dir TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    frame_count INTEGER NOT NULL DEFAULT 0,
    narration_path TEXT
);

CREATE TABLE IF NOT EXISTS draft_jobs (
    job_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'processing',
    draft_json TEXT,
    signal_scores TEXT,
    created_at TEXT NOT NULL
);
"""


def init_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    return conn
