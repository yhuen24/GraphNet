"""
chat_store.py — GraphNet chat persistence layer.

Stores all chat sessions in a single JSON file on disk so conversations
survive page refreshes and server restarts.

File layout  (.graphnet_sessions.json):
{
    "sessions": [
        {
            "id":       "1718000000.123",
            "title":    "First user message (truncated)",
            "created":  "2024-06-10T12:00:00",
            "updated":  "2024-06-10T12:05:00",
            "messages": [
                {
                    "role":          "user" | "assistant",
                    "content":       "...",
                    "msg_id":        "1718000000.123_u",
                    "entities":      [...],
                    "relationships": [...]
                },
                ...
            ]
        },
        ...
    ]
}
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Where sessions are stored. Can be overridden via the env var GRAPHNET_STORE.
DEFAULT_STORE = Path(os.getenv("GRAPHNET_STORE", ".graphnet_sessions.json"))


# ── Low-level I/O ──────────────────────────────────────────────────────────────

def _read_store(path: Path = DEFAULT_STORE) -> Dict[str, Any]:
    """Load the store from disk. Returns empty structure on first run."""
    if not path.exists():
        return {"sessions": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # Corrupt file — start fresh rather than crash.
        return {"sessions": []}


def _write_store(data: Dict[str, Any], path: Path = DEFAULT_STORE) -> None:
    """Atomically write the store to disk (write-then-rename)."""
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except OSError:
        # Best-effort — if we can't write, don't crash the app.
        pass


# ── Public API ─────────────────────────────────────────────────────────────────

def load_all_sessions(path: Path = DEFAULT_STORE) -> List[Dict[str, Any]]:
    """
    Return all sessions from disk, most-recently-updated first.
    Each session dict has: id, title, created, updated, messages.
    """
    data = _read_store(path)
    sessions = data.get("sessions", [])
    # Sort newest-updated first
    sessions.sort(key=lambda s: s.get("updated", ""), reverse=True)
    return sessions


def save_session(session: Dict[str, Any], path: Path = DEFAULT_STORE) -> None:
    """
    Upsert a session by its id.
    Adds 'updated' timestamp automatically.
    """
    data = _read_store(path)
    sessions = data.get("sessions", [])

    session["updated"] = datetime.now().isoformat(timespec="seconds")

    # Replace if exists, otherwise prepend
    for i, s in enumerate(sessions):
        if s["id"] == session["id"]:
            sessions[i] = session
            break
    else:
        sessions.insert(0, session)

    data["sessions"] = sessions
    _write_store(data, path)


def delete_session(session_id: str, path: Path = DEFAULT_STORE) -> None:
    """Remove a session permanently."""
    data = _read_store(path)
    data["sessions"] = [s for s in data["sessions"] if s["id"] != session_id]
    _write_store(data, path)


def new_session(first_message: str = "") -> Dict[str, Any]:
    """
    Create a blank session dict (not yet saved to disk).
    Call save_session() after adding the first message.
    """
    now = datetime.now().isoformat(timespec="seconds")
    title = (first_message[:40] + "…") if len(first_message) > 40 else first_message
    return {
        "id":       str(time.time()),
        "title":    title or "New conversation",
        "created":  now,
        "updated":  now,
        "messages": [],
    }


def get_session(session_id: str, path: Path = DEFAULT_STORE) -> Optional[Dict[str, Any]]:
    """Fetch a single session by id, or None if not found."""
    for s in load_all_sessions(path):
        if s["id"] == session_id:
            return s
    return None


def rename_session(session_id: str, new_title: str,
                   path: Path = DEFAULT_STORE) -> None:
    """Update just the title of a session."""
    data = _read_store(path)
    for s in data["sessions"]:
        if s["id"] == session_id:
            s["title"] = new_title
            s["updated"] = datetime.now().isoformat(timespec="seconds")
            break
    _write_store(data, path)
