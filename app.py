"""
GraphNet - AI Knowledge Graph
Redesigned with a Claude-inspired chat interface.
"""

import streamlit as st
import os
import time
from pathlib import Path
import streamlit.components.v1 as components

from main import GraphNet
from chat_store import (
    load_all_sessions,
    save_session,
    delete_session,
    new_session,
    get_session,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphNet",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (loaded from styles/graphnet.css) ──────────────────────────────
def _load_css(path: str) -> None:
    """Inject a local CSS file into the Streamlit page."""
    with open(path, "r", encoding="utf-8") as fh:
        css = fh.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

_load_css(Path(__file__).parent / "styles" / "graphnet.css")

# JS: (1) measure sidebar width → CSS var, (2) Enter key submits the form
st.markdown("""
<script>
(function() {
    // ── Sidebar width → CSS variable ──────────────────────────────────────
    function setSidebarSize() {
        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        var w = sidebar ? sidebar.getBoundingClientRect().width : 0;
        document.documentElement.style.setProperty('--sidebar-size', w + 'px');
    }
    setSidebarSize();
    window.addEventListener('resize', setSidebarSize);
    var sidebarObserver = new MutationObserver(setSidebarSize);
    sidebarObserver.observe(document.body, { childList: true, subtree: true });

    // ── Enter key → click the send button ─────────────────────────────────
    // Streamlit renders inputs inside iframes sometimes, so we watch for the
    // input appearing and attach the listener each time it is recreated.
    function attachEnterListener() {
        var inputs = document.querySelectorAll('input[type="text"]');
        inputs.forEach(function(input) {
            if (input._enterBound) return;   // already attached
            input._enterBound = true;
            input.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    // Find the send button — it's the button with ↑ label
                    var btns = document.querySelectorAll('button[kind="primary"], button');
                    for (var i = 0; i < btns.length; i++) {
                        if (btns[i].innerText.trim() === '↑') {
                            btns[i].click();
                            break;
                        }
                    }
                }
            });
        });
    }
    // Run on load and whenever DOM changes (Streamlit rerenders wipe listeners)
    attachEnterListener();
    var inputObserver = new MutationObserver(attachEnterListener);
    inputObserver.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)





# ── Session state ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "graphnet":       None,
        "initialized":    False,
        "messages":       [],    # active conversation (in-memory mirror of disk)
        "active_session": None,  # id of the open session
        "show_graph_for": None,  # msg_id whose graph panel is expanded
        "view_mode":      "chat", # "chat" or "graph" — per-message toggle
        "init_attempted": False,
        "input_key":      0,       # incremented to reset the text input widget
        "is_thinking":    False,    # query text while waiting, False when idle
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # On first page load, always start with a blank new chat.
    # Past conversations are accessible via the History section in the sidebar.
    if "sessions_loaded" not in st.session_state:
        st.session_state.sessions_loaded = True

init_state()


# ── Auto-initialize on first load ──────────────────────────────────────────────
def auto_initialize():
    if st.session_state.init_attempted:
        return
    st.session_state.init_attempted = True
    gn = GraphNet()
    result = gn.initialize()
    st.session_state.graphnet = gn
    st.session_state.initialized = result["overall"]


auto_initialize()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _session_from_state() -> dict:
    """Build a session dict from current in-memory state."""
    msgs = st.session_state.messages
    sid  = st.session_state.active_session
    # Derive title from first user message
    title = "New conversation"
    for m in msgs:
        if m["role"] == "user":
            title = (m["content"][:40] + "…") if len(m["content"]) > 40 else m["content"]
            break
    return {"id": sid, "title": title, "messages": list(msgs)}


def save_current_session() -> None:
    """Flush the active in-memory conversation to disk."""
    msgs = st.session_state.messages
    if not msgs:
        return
    sid = st.session_state.active_session
    if not sid:
        # Brand-new session — assign an id now
        sid = str(time.time())
        st.session_state.active_session = sid
    sess = _session_from_state()
    sess["id"] = sid
    save_session(sess)          # chat_store.save_session writes to disk


def start_new_chat() -> None:
    """Save current conversation and open a blank slate."""
    save_current_session()
    st.session_state.messages       = []
    st.session_state.active_session = None
    st.session_state.show_graph_for = None


def load_session_by_id(sid: str) -> None:
    """Save current conversation, then load a different one from disk."""
    save_current_session()
    sess = get_session(sid)         # chat_store.get_session reads from disk
    if sess:
        st.session_state.messages       = list(sess.get("messages", []))
        st.session_state.active_session = sid
        st.session_state.show_graph_for = None


def delete_session_by_id(sid: str) -> None:
    """Delete a session from disk and clear state if it was active."""
    delete_session(sid)             # chat_store.delete_session
    if st.session_state.active_session == sid:
        st.session_state.messages       = []
        st.session_state.active_session = None
        st.session_state.show_graph_for = None


def send_query(query_text):
    """Run the query against GraphNet and append messages."""
    if not query_text.strip():
        return

    # Create a new session id the first time a message is sent
    if not st.session_state.active_session:
        sess = new_session(query_text)   # chat_store.new_session
        st.session_state.active_session = sess["id"]

    msg_id = str(time.time())
    st.session_state.messages.append({
        "role": "user",
        "content": query_text,
        "msg_id": msg_id + "_u",
        "entities": [],
        "relationships": [],
    })

    gn = st.session_state.graphnet
    if not gn or not st.session_state.initialized:
        reply = "⚠️ GraphNet is not initialised. Please check your configuration."
        entities, relationships = [], []
    else:
        result = gn.query(query_text)
        if result.get("success"):
            explanation = result.get("explanation") or "Query executed successfully."
            raw = result.get("results", [])
            # Try to extract entity names from raw results for pills
            entities = []
            for row in raw[:20]:
                for v in row.values():
                    if isinstance(v, dict) and "name" in v:
                        entities.append({
                            "name": v["name"],
                            "type": (v.get("type") or
                                     next(iter(v.get("labels", ["Other"])), "Other")),
                        })
            entities = list({e["name"]: e for e in entities}.values())
            relationships = result.get("results", [])
            reply = explanation
        else:
            reply = f"❌ {result.get('error', 'Query failed.')}"
            entities, relationships = [], []

    ai_msg_id = msg_id + "_a"
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "msg_id": ai_msg_id,
        "entities": entities,
        "relationships": relationships,
    })
    save_current_session()


def process_uploaded_files(files):
    gn = st.session_state.graphnet
    if not gn or not st.session_state.initialized:
        st.error("GraphNet is not initialised.")
        return

    total = len(files)
    bar = st.progress(0, text="Processing…")
    all_entities = []

    for i, f in enumerate(files):
        bar.progress((i) / total, text=f"Processing {f.name}…")
        result = gn.process_document(
            file_bytes=f.read(),
            file_extension=Path(f.name).suffix,
            filename=f.name,
        )
        if result.get("success"):
            all_entities.append(
                f"**{f.name}** — {result['entities_extracted']} entities, "
                f"{result['relationships_extracted']} relationships extracted."
            )
        else:
            all_entities.append(f"**{f.name}** — ❌ {result.get('error', 'failed')}")
        bar.progress((i + 1) / total, text=f"Done: {f.name}")

    bar.empty()

    summary = "\n".join(f"• {line}" for line in all_entities)
    msg_id = str(time.time())
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"📂 **Files processed:**\n{summary}\n\nYou can now ask questions about this data.",
        "msg_id": msg_id,
        "entities": [],
        "relationships": [],
    })
    save_current_session()


def render_graph(msg):
    """Inline graph visualisation for a given message's entities/relationships."""
    gn = st.session_state.graphnet
    if not gn:
        st.warning("Not initialised.")
        return

    entities = msg.get("entities", [])
    rels = msg.get("relationships", [])

    if not entities and not rels:
        # Fall back to fetching full graph
        try:
            fname = gn.visualize_graph(limit=80)
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as fh:
                    components.html(fh.read(), height=620, scrolling=True)
            else:
                st.info("No graph data available yet.")
        except Exception as e:
            st.error(f"Visualisation error: {e}")
        return

    # Build a mini vis with pyvis directly
    try:
        from pyvis.network import Network
        net = Network(height="600px", width="100%", bgcolor="#17171a",
                      font_color="#e8e8ec", directed=True)
        net.set_options("""{
          "physics":{"enabled":true,"stabilization":{"iterations":120}},
          "interaction":{"hover":true,"navigationButtons":true,"zoomView":true},
          "edges":{"color":{"color":"#aaaaaa","highlight":"#ffffff","hover":"#ffffff"},"width":1.5,"smooth":{"type":"dynamic"}}
        }""")

        COLORS = {
            "Person": "#a89cf9", "Organization": "#7df3e4",
            "Location": "#fbbf24", "Concept": "#f472b6",
            "Product": "#4ade80", "Technology": "#60a5fa",
        }
        added = set()
        for e in entities:
            nid = e["name"]
            if nid not in added:
                color = COLORS.get(e.get("type", "Other"), "#9b9baa")
                net.add_node(nid, label=nid, color=color, size=22,
                             title=f"{e.get('type','?')}: {nid}")
                added.add(nid)

        for r in rels[:30]:
            s = r.get("source") or r.get("e1", {})
            t = r.get("target") or r.get("e2", {})
            rel_type = r.get("relationship") or r.get("type", "RELATED")
            if isinstance(s, dict): s = s.get("name", "")
            if isinstance(t, dict): t = t.get("name", "")
            if s and t:
                for nid in [s, t]:
                    if nid not in added:
                        net.add_node(nid, label=nid, color="#9b9baa", size=18)
                        added.add(nid)
                net.add_edge(s, t, label=str(rel_type),
                             color={"color": "#cccccc", "highlight": "#ffffff", "hover": "#ffffff"},
                             width=1.5, arrows="to")

        html_str = net.generate_html()
        # scrolling=True lets the user scroll within the graph iframe
        components.html(html_str, height=620, scrolling=True)

    except Exception as e:
        st.error(f"Visualisation error: {e}")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <span class="icon">🕸️</span>
        <span class="name">GraphNet</span>
    </div>""", unsafe_allow_html=True)

    # New chat button
    if st.button("＋  New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    # Upload section
    st.markdown('<div class="sidebar-section-label">Upload documents</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "xlsx", "pptx", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("Process files", use_container_width=True):
            process_uploaded_files(uploaded)
            st.rerun()

    # Chat history (loaded fresh from disk on every render)
    all_sessions = load_all_sessions()
    if all_sessions:
        st.markdown('<div class="sidebar-section-label">History</div>',
                    unsafe_allow_html=True)
        for s in all_sessions[:30]:
            label = s["title"][:34] + ("…" if len(s["title"]) > 34 else "")
            is_active = s["id"] == st.session_state.active_session
            # Row: load button + delete button
            col_l, col_d = st.columns([9, 1])
            with col_l:
                btn_label = f"{'▶ ' if is_active else '💬  '}{label}"
                if st.button(btn_label, key=f"hist_{s['id']}",
                             use_container_width=True):
                    load_session_by_id(s["id"])
                    st.rerun()
            with col_d:
                if st.button("✕", key=f"del_{s['id']}"):
                    delete_session_by_id(s["id"])
                    st.rerun()

    # Connection status at bottom
    st.markdown("---")
    if st.session_state.initialized:
        st.markdown(
            '<span class="status-dot"></span>'
            '<span style="font-size:0.75rem;color:#6b6b78;">Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-dot status-offline"></span>'
            '<span style="font-size:0.75rem;color:#6b6b78;">Not connected</span>',
            unsafe_allow_html=True,
        )
        if st.button("Retry connection", use_container_width=True):
            gn = GraphNet()
            result = gn.initialize()
            st.session_state.graphnet = gn
            st.session_state.initialized = result["overall"]
            st.rerun()


# ── Top bar ────────────────────────────────────────────────────────────────────
_status = (
    '<span class="status-dot"></span> Ready'
    if st.session_state.initialized
    else '<span class="status-dot status-offline"></span> Initialising…'
)
st.markdown(f'''
<div class="chat-top-bar">
    <h2>GraphNet</h2>
    <div style="font-size:0.78rem;color:var(--muted);">{_status}</div>
</div>
''', unsafe_allow_html=True)

messages = st.session_state.messages


# ── Message renderer ──────────────────────────────────────────────────────────────────────────────
def render_messages():
    if not messages:
        st.markdown("""
        <div class="welcome">
            <div class="welcome-icon">🕸️</div>
            <h1>What do you want to know?</h1>
            <p>Upload documents and ask questions.<br>
               GraphNet surfaces entities, relationships, and insights from your data.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    for msg in messages:
        role     = msg["role"]
        content  = msg["content"]
        mid      = msg.get("msg_id", "")
        entities = msg.get("entities", [])

        if role == "user":
            st.markdown(f"""
            <div class="message-row user">
                <div class="avatar user">You</div>
                <div><div class="bubble user">{content}</div></div>
            </div>""", unsafe_allow_html=True)

        else:
            # --- per-message view state ---
            view_key = f"view_{mid}"
            if view_key not in st.session_state:
                st.session_state[view_key] = "chat"
            is_graph = st.session_state[view_key] == "graph"

            # Toggle row — ALWAYS rendered first, before any content,
            # so it is never pushed off-screen by a tall graph panel.
            st.markdown('<div class="toggle-row">', unsafe_allow_html=True)
            tc1, tc2, _ = st.columns([1, 1, 10])
            with tc1:
                if st.button(
                    "💬 Answer",
                    key=f"chat_tab_{mid}",
                    type="primary" if not is_graph else "secondary",
                ):
                    st.session_state[view_key] = "chat"
                    st.rerun()
            with tc2:
                if st.button(
                    "⬡ Graph",
                    key=f"graph_tab_{mid}",
                    type="primary" if is_graph else "secondary",
                ):
                    st.session_state[view_key] = "graph"
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            # Build entity pills HTML
            pills_html = ""
            if entities:
                pills_html = '<div class="entity-pills">'
                for e in entities[:12]:
                    etype = e.get("type", "Other")
                    pills_html += f'<span class="entity-pill pill-{etype}">{e["name"]}</span>'
                pills_html += "</div>"

            if not is_graph:
                # Chat answer view
                st.markdown(f"""
                <div class="message-row">
                    <div class="avatar ai">GN</div>
                    <div style="flex:1;min-width:0">
                        <div class="bubble ai">{content}{pills_html}</div>
                        <div class="bubble-meta">GraphNet</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                # Graph view
                st.markdown('<div class="message-row"><div class="avatar ai">GN</div><div style="flex:1;min-width:0">', unsafe_allow_html=True)
                st.markdown('<div class="graph-panel">', unsafe_allow_html=True)
                render_graph(msg)
                st.markdown('</div></div></div>', unsafe_allow_html=True)


# Phase 2 — process stored query now that thinking indicator has rendered
if st.session_state.is_thinking:
    _q = st.session_state.is_thinking
    st.session_state.is_thinking = False
    send_query(_q)
    st.rerun()

st.markdown('<div class="messages-scroll">', unsafe_allow_html=True)
render_messages()
# Show animated thinking indicator while waiting for LLM response
if st.session_state.is_thinking:
    st.markdown('<div class="thinking-row"><div class="avatar ai">GN</div><div class="thinking-bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Input bar ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="input-bar"><div class="input-bar-inner">', unsafe_allow_html=True)

_, inner_left, col_input, col_btn, inner_right, _ = st.columns([1, 0.05, 6, 0.5, 0.05, 1])
with col_input:
    query = st.text_input(
        "query",
        placeholder="Ask anything about your documents…",
        label_visibility="collapsed",
        key=f"chat_input_{st.session_state.input_key}",
    )
with col_btn:
    send = st.button("↑", key="send_btn")

st.markdown("</div></div>", unsafe_allow_html=True)

# Send on button click OR Enter (Enter is handled by JS which clicks the button)
if send and query and not st.session_state.is_thinking:
    st.session_state.is_thinking = query
    st.session_state.input_key += 1
    st.rerun()