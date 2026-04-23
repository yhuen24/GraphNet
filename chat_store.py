"""
chat_store.py — GraphNet chat persistence layer (session-scoped).

Stores all chat sessions in Streamlit's per-browser session_state so that
each visitor gets their own isolated chat history. Chats will not persist
across page refreshes, but — critically — users can no longer see each
other's conversations.

The public API is identical to the previous disk-backed version, so
app.py requires NO changes.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_store() -> List[Dict[str, Any]]:
    """Return the sessions list from session_state, creating it if needed."""
    if "_chat_sessions" not in st.session_state:
        st.session_state["_chat_sessions"] = []
    return st.session_state["_chat_sessions"]


# ── Public API (same signatures as the old disk-backed version) ────────────────

def load_all_sessions(path=None) -> List[Dict[str, Any]]:
    """
    Return all sessions, most-recently-updated first.
    `path` is accepted but ignored (kept for API compatibility).
    """
    sessions = list(_get_store())
    sessions.sort(key=lambda s: s.get("updated", ""), reverse=True)
    return sessions


def save_session(session: Dict[str, Any], path=None) -> None:
    """Upsert a session by its id."""
    sessions = _get_store()
    session["updated"] = datetime.now().isoformat(timespec="seconds")

    if "created" not in session:
        session["created"] = session["updated"]

    for i, s in enumerate(sessions):
        if s["id"] == session["id"]:
            sessions[i] = session
            return
    sessions.insert(0, session)


def delete_session(session_id: str, path=None) -> None:
    """Remove a session permanently."""
    store = _get_store()
    st.session_state["_chat_sessions"] = [
        s for s in store if s["id"] != session_id
    ]


def new_session(first_message: str = "") -> Dict[str, Any]:
    """
    Create a blank session dict (not yet saved).
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
    for s in _get_store():
        if s["id"] == session_id:
            return s
    return None


def rename_session(session_id: str, new_title: str, path=None) -> None:
    """Update just the title of a session."""
    for s in _get_store():
        if s["id"] == session_id:
            s["title"] = new_title
            s["updated"] = datetime.now().isoformat(timespec="seconds")
            return