"""
test_chat_store.py — Tests for chat_store.py

Covers: session creation, persistence across load/save cycles,
atomic write safety, corrupt file recovery, and session ordering.
"""

import pytest
import json
import time
from pathlib import Path

from chat_store import (
    _read_store,
    _write_store,
    load_all_sessions,
    save_session,
)


# ═══════════════════════════════════════════════════════════════════════════
# Low-level I/O
# ═══════════════════════════════════════════════════════════════════════════

class TestStoreIO:
    """Tests for _read_store and _write_store."""

    def test_read_nonexistent_returns_empty(self, tmp_dir):
        path = tmp_dir / "nonexistent.json"
        data = _read_store(path)
        assert data == {"sessions": []}

    def test_write_then_read_roundtrip(self, tmp_dir):
        path = tmp_dir / "store.json"
        original = {"sessions": [{"id": "1", "title": "Test"}]}
        _write_store(original, path)
        loaded = _read_store(path)
        assert loaded == original

    def test_corrupt_file_returns_empty(self, tmp_dir):
        path = tmp_dir / "corrupt.json"
        path.write_text("{invalid json!!", encoding="utf-8")
        data = _read_store(path)
        assert data == {"sessions": []}

    def test_atomic_write_leaves_no_tmp_file(self, tmp_dir):
        path = tmp_dir / "store.json"
        _write_store({"sessions": []}, path)
        tmp_file = path.with_suffix(".tmp")
        assert not tmp_file.exists()
        assert path.exists()


# ═══════════════════════════════════════════════════════════════════════════
# Session CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionOperations:
    """Tests for save_session and load_all_sessions."""

    def _make_session(self, session_id, title="Test", messages=None):
        return {
            "id": session_id,
            "title": title,
            "created": "2024-06-10T12:00:00",
            "updated": "2024-06-10T12:00:00",
            "messages": messages or [],
        }

    def test_save_new_session(self, tmp_dir):
        path = tmp_dir / "store.json"
        session = self._make_session("s1", "First session")
        save_session(session, path)

        sessions = load_all_sessions(path)
        assert len(sessions) == 1
        assert sessions[0]["id"] == "s1"
        assert sessions[0]["title"] == "First session"

    def test_save_updates_existing_session(self, tmp_dir):
        path = tmp_dir / "store.json"
        session = self._make_session("s1", "Original")
        save_session(session, path)

        session["title"] = "Updated"
        session["messages"] = [{"role": "user", "content": "Hello"}]
        save_session(session, path)

        sessions = load_all_sessions(path)
        assert len(sessions) == 1
        assert sessions[0]["title"] == "Updated"
        assert len(sessions[0]["messages"]) == 1

    def test_multiple_sessions_persist(self, tmp_dir):
        path = tmp_dir / "store.json"
        save_session(self._make_session("s1", "First"), path)
        save_session(self._make_session("s2", "Second"), path)
        save_session(self._make_session("s3", "Third"), path)

        sessions = load_all_sessions(path)
        assert len(sessions) == 3
        ids = {s["id"] for s in sessions}
        assert ids == {"s1", "s2", "s3"}

    def test_sessions_sorted_newest_first(self, tmp_dir):
        path = tmp_dir / "store.json"

        s1 = self._make_session("s1")
        s1["updated"] = "2024-01-01T00:00:00"

        s2 = self._make_session("s2")
        s2["updated"] = "2024-06-15T00:00:00"

        s3 = self._make_session("s3")
        s3["updated"] = "2024-03-10T00:00:00"

        # Write directly to avoid save_session overwriting 'updated'
        _write_store({"sessions": [s1, s2, s3]}, path)

        sessions = load_all_sessions(path)
        assert sessions[0]["id"] == "s2"  # most recent
        assert sessions[-1]["id"] == "s1"  # oldest

    def test_message_with_entities_persists(self, tmp_dir):
        path = tmp_dir / "store.json"
        session = self._make_session("s1")
        session["messages"] = [
            {
                "role": "assistant",
                "content": "HSBC is headquartered in London.",
                "msg_id": "s1_a1",
                "entities": [
                    {"name": "HSBC", "type": "Organization"},
                    {"name": "London", "type": "Location"},
                ],
                "relationships": [
                    {"source": "HSBC", "target": "London",
                     "type": "HEADQUARTERED_IN"},
                ],
            }
        ]
        save_session(session, path)

        sessions = load_all_sessions(path)
        msg = sessions[0]["messages"][0]
        assert len(msg["entities"]) == 2
        assert len(msg["relationships"]) == 1
        assert msg["entities"][0]["name"] == "HSBC"

    def test_unicode_content_preserved(self, tmp_dir):
        path = tmp_dir / "store.json"
        session = self._make_session("s1")
        session["messages"] = [
            {"role": "user", "content": "日本語テスト — café — ñoño"}
        ]
        save_session(session, path)

        sessions = load_all_sessions(path)
        assert "café" in sessions[0]["messages"][0]["content"]
        assert "ñoño" in sessions[0]["messages"][0]["content"]

    def test_empty_store_returns_empty_list(self, tmp_dir):
        path = tmp_dir / "store.json"
        sessions = load_all_sessions(path)
        assert sessions == []