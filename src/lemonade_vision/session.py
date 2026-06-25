import uuid
import sqlite3
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expiry_iso(ttl_seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()


def create_session(db: sqlite3.Connection, tmp_dir: str, ttl_seconds: int = 600) -> str:
    session_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO capture_sessions (session_id, tmp_dir, expires_at) VALUES (?, ?, ?)",
        (session_id, tmp_dir, _expiry_iso(ttl_seconds)),
    )
    db.commit()
    return session_id


def validate_session(db: sqlite3.Connection, session_id: str) -> Optional[sqlite3.Row]:
    row = db.execute(
        "SELECT * FROM capture_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row is None:
        return None
    if row["expires_at"] < _now_iso():
        _cleanup_session(db, dict(row))
        return None
    return row


def close_session(db: sqlite3.Connection, session_id: str) -> None:
    row = db.execute(
        "SELECT * FROM capture_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row:
        _cleanup_session(db, dict(row))


def expire_old_sessions(db: sqlite3.Connection) -> int:
    rows = db.execute(
        "SELECT * FROM capture_sessions WHERE expires_at < ?", (_now_iso(),)
    ).fetchall()
    for row in rows:
        _cleanup_session(db, dict(row))
    return len(rows)


def _cleanup_session(db: sqlite3.Connection, row: dict) -> None:
    tmp = Path(row["tmp_dir"])
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    db.execute("DELETE FROM capture_sessions WHERE session_id = ?", (row["session_id"],))
    db.commit()
