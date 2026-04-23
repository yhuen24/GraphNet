"""
chat_store.py — GraphNet chat persistence layer (per-user).

Each visitor is assigned a unique ID (stored in the browser URL via
st.query_params).  Sessions are persisted to a user-scoped JSON file
on disk, so chats survive page refreshes while remaining invisible
to other users.

File per user:  .graphnet_sessions_{uid}.json

The public API is identical to the original version — app.py needs
only one small addition in init_state() to call ensure_user_id().
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

# Base directory for all session files
STORE_DIR = Path(os.getenv("GRAPHNET_STORE_DIR", "."))


# ── User identity ──────────────────────────────────────────────────────────────

def ensure_user_id() -> str:
    """
    Ensure the current browser tab has a unique user ID.
    Call this once at app startup (e.g. inside init_state()).

    The ID is kept in st.query_params so it persists across
    refreshes for the same browser tab/bookmark.
    """
    uid = st.query_params.get("uid")
    if not uid:
        uid = uuid.uuid4().hex[:12]
        st.query_params["uid"] = uid
    # Also cache in session_state for fast access
    st.session_state["_graphnet_uid"] = uid
    return uid


def _user_store_path() -> Path:
    """Return the JSON file path scoped to the current user."""
    uid = st.session_state.get("_graphnet_uid", "default")
    return STORE_DIR / f".graphnet_sessions_{uid}.json"


# ── Low-level I/O ──────────────────────────────────────────────────────────────

def _read_store() -> Dict[str, Any]:
    """Load the user's store from disk."""
    path = _user_store_path()
    if not path.exists():
        return {"sessions": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"sessions": []}


def _write_store(data: Dict[str, Any]) -> None:
    """Atomically write the user's store to disk."""
    path = _user_store_path()
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except OSError:
        pass


# ── Public API (unchanged signatures) ─────────────────────────────────────────

def load_all_sessions(path=None) -> List[Dict[str, Any]]:
    """
    Return all sessions for the current user, most-recently-updated first.
    `path` parameter kept for API compat but ignored.
    """
    data = _read_store()
    sessions = data.get("sessions", [])
    sessions.sort(key=lambda s: s.get("updated", ""), reverse=True)
    return sessions


def save_session(session: Dict[str, Any], path=None) -> None:
    """Upsert a session by its id."""
    data = _read_store()
    sessions = data.get("sessions", [])

    session["updated"] = datetime.now().isoformat(timespec="seconds")
    if "created" not in session:
        session["created"] = session["updated"]

    for i, s in enumerate(sessions):
        if s["id"] == session["id"]:
            sessions[i] = session
            break
    else:
        sessions.insert(0, session)

    data["sessions"] = sessions
    _write_store(data)


def delete_session(session_id: str, path=None) -> None:
    """Remove a session permanently."""
    data = _read_store()
    data["sessions"] = [s for s in data["sessions"] if s["id"] != session_id]
    _write_store(data)


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


def get_session(session_id: str, path=None) -> Optional[Dict[str, Any]]:
    """Fetch a single session by id, or None if not found."""
    for s in load_all_sessions():
        if s["id"] == session_id:
            return s
    return None


def rename_session(session_id: str, new_title: str, path=None) -> None:
    """Update just the title of a session."""
    data = _read_store()
    for s in data["sessions"]:
        if s["id"] == session_id:
            s["title"] = new_title
            s["updated"] = datetime.now().isoformat(timespec="seconds")
            break
    _write_store(data)