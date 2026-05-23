import tempfile
import time
from lemonade_vision.store.schema import init_db
from lemonade_vision.session import (
    create_session, validate_session, close_session, expire_old_sessions,
)


def _db():
    tmp = tempfile.mktemp(suffix=".db", dir="/tmp")
    db = init_db(tmp)
    return db


def test_create_and_validate_session():
    db = _db()
    tmp_dir = tempfile.mkdtemp(dir="/tmp")
    sid = create_session(db, tmp_dir, ttl_seconds=300)
    assert sid is not None
    result = validate_session(db, sid)
    assert result is not None
    assert result["session_id"] == sid


def test_expired_session_returns_none():
    db = _db()
    tmp_dir = tempfile.mkdtemp(dir="/tmp")
    sid = create_session(db, tmp_dir, ttl_seconds=0)
    time.sleep(0.1)
    result = validate_session(db, sid)
    assert result is None


def test_close_session_cleans_up():
    db = _db()
    tmp_dir = tempfile.mkdtemp(dir="/tmp")
    sid = create_session(db, tmp_dir, ttl_seconds=300)
    close_session(db, sid)
    result = validate_session(db, sid)
    assert result is None


def test_expire_old_sessions_removes_expired():
    db = _db()
    tmp_dir = tempfile.mkdtemp(dir="/tmp")
    sid = create_session(db, tmp_dir, ttl_seconds=0)
    time.sleep(0.1)
    count = expire_old_sessions(db)
    assert count >= 1
    assert validate_session(db, sid) is None
