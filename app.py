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
    ensure_user_id,
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
    ensure_user_id()
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


# ── Helper: safely get a string from a value that might be str, dict, or list ─
def _safe_str(val):
    """Return a display string from a result value (str, dict, or list)."""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("name", "")
    if isinstance(val, list) and val:
        return str(val[0])
    return ""


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
        entities, relationships, focal_entities = [], [], []
    else:
        result = gn.query(query_text)
        focal_entities = result.get("focal_entities", []) if isinstance(result, dict) else []
        if result.get("success"):
            explanation = result.get("explanation") or "Query executed successfully."
            raw = result.get("results", [])



            entities = []
            seen_names = set()

            # Keys that typically hold entity names
            NAME_KEYS = {
                "entity", "name", "connected_entity",
                "n.name", "m.name",
            }
            # Keys that hold the corresponding type / labels
            TYPE_KEYS = {
                "entity_type", "connected_type", "type", "labels",
            }

            for row in raw[:30]:
                # Collect all type/label values in this row for lookup
                row_types = {}
                for k, v in row.items():
                    if k in TYPE_KEYS or k.endswith("_type") or k == "labels":
                        row_types[k] = v

                # --- string-valued name fields ---
                for k, v in row.items():
                    if k not in NAME_KEYS:
                        continue
                    name = _safe_str(v)
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)

                    # Resolve entity type
                    etype = "Other"
                    # Try partner key first (entity→entity_type, connected_entity→connected_type)
                    partner = k.replace("entity", "type").replace("name", "type")
                    raw_type = (row_types.get(partner)
                                or row_types.get("entity_type")
                                or row_types.get("labels")
                                or row_types.get("type"))
                    if raw_type:
                        if isinstance(raw_type, list) and raw_type:
                            etype = str(raw_type[0])
                        elif isinstance(raw_type, str):
                            etype = raw_type

                    entities.append({"name": name, "type": etype})

                # --- dict-valued results (Neo4j node objects) ---
                for k, v in row.items():
                    if isinstance(v, dict) and "name" in v:
                        name = v["name"]
                        if name in seen_names:
                            continue
                        seen_names.add(name)
                        etype = (v.get("type")
                                 or next(iter(v.get("labels", ["Other"])), "Other"))
                        entities.append({"name": name, "type": etype})

            # ── Extract relationships ─────────────────────────────────
            relationships = []
            for row in raw[:30]:
                src = _safe_str(
                    row.get("entity")
                    or row.get("name")
                    or row.get("n", "")
                )
                tgt = _safe_str(
                    row.get("connected_entity")
                    or row.get("m", "")
                )
                rel = _safe_str(
                    row.get("relationship")
                    or row.get("rel_type")
                    or row.get("type", "")
                ) or "RELATED"

                if src and tgt:
                    relationships.append({
                        "source": src,
                        "target": tgt,
                        "type": rel,
                    })

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
        "focal_entities": focal_entities,
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
    focal = msg.get("focal_entities", [])

    # ── If no focal entities, derive them from query-result entities ──
    # This prevents falling back to the full graph dump.
    if not focal and entities:
        focal = [e["name"] for e in entities[:15]]  # Cap at 15 focal seeds

    # ── Focused graph: only show entities + connections from the answer ──
    # Fetches relationships from Neo4j but ONLY keeps connections where
    # the connected entity is also mentioned in the answer text.
    # This ensures the graph mirrors exactly what the text discusses.
    MAX_CONNECTED_PER_FOCAL = 8  # Prevent any single entity from sprawling

    if focal and hasattr(gn, 'graph_manager'):
        gm = gn.graph_manager
        entities = []
        rels = []
        seen_names = set()
        seen_edges = set()  # Deduplicate edges by (source, target, type)
        answer_lower = msg.get("content", "").lower()
        focal_lower_set = {f.strip().lower() for f in focal}

        # Only expand relationships for the PRIMARY query targets (first 2).
        # Other focal entities appear as leaf nodes only if they're directly
        # connected to a primary target — this keeps the graph at 1-hop.
        PRIMARY_FOCAL_LIMIT = 2
        primary_focal = focal[:PRIMARY_FOCAL_LIMIT]
        secondary_focal = focal[PRIMARY_FOCAL_LIMIT:]

        # Add ALL focal entities as nodes (but only primary ones get expanded)
        for focal_name in focal:
            if focal_name not in seen_names:
                seen_names.add(focal_name)
                entity_data = gm.get_entity(focal_name)
                etype = "Other"
                if entity_data:
                    etype = (entity_data.get("type")
                             or next(iter(entity_data.get("labels", ["Other"])), "Other"))
                entities.append({"name": focal_name, "type": etype})

        # Only fetch and expand relationships for primary focal entities
        for focal_name in primary_focal:
            connected_count = 0
            try:
                relationships = gm.get_entity_relationships(focal_name)
                for row in relationships:
                    connected = row.get("entity", "")
                    rel_type = row.get("relationship", "RELATED")
                    labels = row.get("labels", [])

                    if not connected:
                        continue

                    # FILTER: only keep if the connected entity appears in
                    # the answer text OR is another focal entity
                    connected_lower = connected.strip().lower()
                    in_answer = connected_lower in answer_lower
                    is_focal = connected_lower in focal_lower_set

                    if not in_answer and not is_focal:
                        continue

                    # Limit sprawl per focal entity
                    if connected_count >= MAX_CONNECTED_PER_FOCAL:
                        break

                    if connected not in seen_names:
                        seen_names.add(connected)
                        ctype = str(labels[0]) if labels else "Other"
                        entities.append({"name": connected, "type": ctype})

                    # Use actual relationship direction from Neo4j
                    actual_source = row.get("rel_source", focal_name)
                    actual_target = row.get("rel_target", connected)

                    edge_key = (actual_source, actual_target, rel_type)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        rels.append({
                            "source": actual_source,
                            "target": actual_target,
                            "type": rel_type,
                        })
                    connected_count += 1
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Could not fetch relationships for '{focal_name}': {e}"
                )

        # Remove entities that ended up with no connections (isolated nodes)
        # Primary focal entities are always kept even if isolated.
        primary_lower = {f.strip().lower() for f in primary_focal}
        connected_names = set()
        for r in rels:
            connected_names.add(r["source"])
            connected_names.add(r["target"])
        entities = [
            e for e in entities
            if e["name"] in connected_names
            or e["name"].strip().lower() in primary_lower
        ]

    if not entities and not rels:
        st.info("No graph data for this query. Try asking about specific entities or topics.")
        return

    # Build a mini vis with pyvis directly
    try:
        import json as _json
        from pyvis.network import Network
        net = Network(height="600px", width="100%", bgcolor="#17171a",
                      font_color="#e8e8ec", directed=True)
        net.set_options("""{
          "physics":{
            "enabled":true,
            "barnesHut":{
              "gravitationalConstant":-15000,
              "centralGravity":0.25,
              "springLength":350,
              "springConstant":0.02,
              "damping":0.15,
              "avoidOverlap":0.6
            },
            "stabilization":{"iterations":200}
          },
          "interaction":{"hover":true,"navigationButtons":true,"zoomView":true},
          "edges":{
            "color":{"color":"#5eead4","highlight":"#99f6e4","hover":"#99f6e4","inherit":false},
            "width":1.8,
            "arrows":{"to":{"enabled":true,"scaleFactor":0.8}},
            "font":{"color":"#b8b8c0","size":9,"strokeWidth":2,"strokeColor":"#17171a"},
            "smooth":{"type":"continuous"}
          }
        }""")

        COLORS = {
            "Person": "#a89cf9", "Organization": "#7df3e4",
            "Location": "#fbbf24", "Concept": "#f472b6",
            "Product": "#4ade80", "Technology": "#60a5fa",
            "Document": "#a78bfa", "Project": "#38bdf8",
            "Task": "#fb923c", "Role": "#f9a8d4",
            "Department": "#34d399", "Event": "#c084fc",
            "Date": "#fcd34d", "Skill": "#67e8f9",
            "Policy": "#fda4af", "Risk": "#f87171",
            "Decision": "#fdba74", "Regulation": "#d8b4fe",
            "Contract": "#86efac", "Resource": "#93c5fd",
            "Deliverable": "#fca5a5", "Qualification": "#a5f3fc",
            "Metric": "#d9f99d", "Course": "#e9d5ff",
            "Standard": "#bfdbfe",
        }
        DEFAULT_COLOR = "#9b9baa"
        added = set()

        # Build node metadata for the click panel
        # Fetch descriptions and sources from the graph for each entity
        node_meta = {}

        # Pre-build a name→node lookup from the graph for reliable type resolution
        _name_lookup = {}
        if hasattr(gn.graph_manager, 'graph'):
            for _nid in gn.graph_manager.graph.nodes():
                _nd = gn.graph_manager.graph.nodes[_nid]
                _n = _nd.get("name", "").strip().lower()
                if _n:
                    _name_lookup[_n] = _nd

        def _humanize_rel(rel_type):
            """WORKS_FOR → 'works for', LOCATED_IN → 'located in'"""
            return rel_type.replace("_", " ").lower()

        def _fetch_entity_info(name):
            """Pull description, source, type, AND all properties from the graph."""
            info = {"description": "", "source": "", "type": "", "properties": {}}
            # Keys to exclude from the properties display
            SKIP_KEYS = {"name", "canonical_name", "created", "updated",
                         "description", "source", "type", "confidence"}
            try:
                entity_data = gn.graph_manager.get_entity(name)
                if entity_data:
                    info["description"] = entity_data.get("description", "")
                    info["source"] = entity_data.get("source", "")
                    info["type"] = entity_data.get("type", "")
                    # Grab ALL remaining properties (value, unit, period, etc.)
                    for k, v in entity_data.items():
                        if k not in SKIP_KEYS and v not in (None, "", [], {}):
                            info["properties"][k] = v

                # Fallback: direct graph node lookup by name (embedded mode)
                if not info["type"]:
                    nd = _name_lookup.get(name.strip().lower())
                    if nd:
                        info["type"] = nd.get("type", "")
                        if not info["description"]:
                            info["description"] = nd.get("description", "")
                        if not info["source"]:
                            info["source"] = nd.get("source", "")
                        if not info["properties"]:
                            for k, v in nd.items():
                                if k not in SKIP_KEYS and v not in (None, "", [], {}):
                                    info["properties"][k] = v
            except Exception:
                pass
            return info

        # Add entity nodes
        for e in entities:
            nid = e["name"]
            if nid not in added:
                etype = e.get("type", "Other")
                info = _fetch_entity_info(nid)
                # If the query result gave a vague type, use the real one from the graph
                if etype in ("Other", "Unknown", "") and info["type"]:
                    etype = info["type"]
                color = COLORS.get(etype, DEFAULT_COLOR)

                # Focal nodes are larger with a gold border (star center)
                is_focal = focal and nid.strip().lower() in {f.strip().lower() for f in focal}
                node_size = 35 if is_focal else 22
                border_color = "#FFD700" if is_focal else color
                node_opts = dict(
                    label=nid, color={"background": color,
                                       "border": border_color,
                                       "highlight": {"background": color,
                                                     "border": "#FFD700"}},
                    size=node_size,
                    borderWidth=3 if is_focal else 1,
                    title=f"{etype}: {nid}",
                )
                net.add_node(nid, **node_opts)
                added.add(nid)
                node_meta[nid] = {
                    "type": etype,
                    "color": color,
                    "description": info["description"],
                    "source": info["source"],
                    "properties": info["properties"],
                    "connections": [],
                }

        # Add relationship edges + collect connection metadata
        for r in rels[:30]:
            s = r.get("source", "")
            t = r.get("target", "")
            rel_type = r.get("type", "RELATED")

            if not s or not t:
                continue

            for nid in [s, t]:
                if nid not in added:
                    info = _fetch_entity_info(nid)
                    # Use the REAL type from the graph, not hardcoded "Unknown"
                    real_type = info["type"] or "Other"
                    color = COLORS.get(real_type, DEFAULT_COLOR)
                    net.add_node(nid, label=nid, color=color, size=18)
                    added.add(nid)
                    node_meta[nid] = {
                        "type": real_type,
                        "color": color,
                        "description": info["description"],
                        "source": info["source"],
                        "connections": [],
                    }

            net.add_edge(s, t, label=str(rel_type),
                         width=2, arrows="to")

            # Build readable sentences for connections
            readable = _humanize_rel(rel_type)
            if s in node_meta:
                node_meta[s]["connections"].append({
                    "sentence": f"{readable} {t}",
                })
            if t in node_meta:
                node_meta[t]["connections"].append({
                    "sentence": f"{s} {readable} this entity",
                })

        html_str = net.generate_html()

        # ── Inject click handler + info panel ────────────────────────
        meta_json = _json.dumps(node_meta, ensure_ascii=False)

        injection = """
<style>
#nodeInfoPanel {
  display: none;
  position: absolute;
  top: 12px;
  right: 12px;
  width: 300px;
  max-height: calc(100% - 24px);
  overflow-y: auto;
  background: #1e1e23;
  border: 1px solid #2a2a30;
  border-radius: 12px;
  padding: 18px;
  font-family: 'DM Sans', -apple-system, sans-serif;
  color: #e8e8ec;
  z-index: 1000;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
#nodeInfoPanel .panel-close {
  position: absolute;
  top: 8px;
  right: 12px;
  cursor: pointer;
  color: #6b6b78;
  font-size: 18px;
  line-height: 1;
  border: none;
  background: none;
}
#nodeInfoPanel .panel-close:hover { color: #e8e8ec; }
#nodeInfoPanel .node-name {
  font-size: 1.05rem;
  font-weight: 600;
  margin-bottom: 6px;
  padding-right: 24px;
  line-height: 1.3;
}
#nodeInfoPanel .node-type {
  display: inline-block;
  font-size: 0.7rem;
  padding: 2px 10px;
  border-radius: 20px;
  font-weight: 500;
  margin-bottom: 14px;
}
#nodeInfoPanel .node-desc {
  font-size: 0.82rem;
  line-height: 1.6;
  color: #c8c8d0;
  margin-bottom: 14px;
}
#nodeInfoPanel .section-label {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #6b6b78;
  margin: 14px 0 8px;
}
#nodeInfoPanel .conn-sentence {
  font-size: 0.8rem;
  line-height: 1.55;
  color: #c8c8d0;
  padding: 3px 0;
}
#nodeInfoPanel .conn-sentence .conn-highlight {
  color: #5eead4;
  font-weight: 500;
}
#nodeInfoPanel .no-info {
  color: #6b6b78;
  font-size: 0.8rem;
  font-style: italic;
}
#nodeInfoPanel .source-tag {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid #2a2a30;
  font-size: 0.75rem;
  color: #6b6b78;
}
#nodeInfoPanel .source-tag .source-icon { font-size: 0.85rem; }
#nodeInfoPanel .source-tag .source-name { color: #a8a8b4; }
#nodeInfoPanel .click-hint {
  color: #4a4a56;
  font-size: 0.68rem;
  text-align: center;
  margin-top: 10px;
}
</style>

<div id="nodeInfoPanel"></div>

<script>
(function() {
  var meta = """ + meta_json + """;

  var pillColors = {
    "Person":       {bg:"#7c6af722", fg:"#a89cf9", bd:"#7c6af740"},
    "Organization": {bg:"#5eead422", fg:"#7df3e4", bd:"#5eead440"},
    "Location":     {bg:"#f59e0b22", fg:"#fbbf24", bd:"#f59e0b40"},
    "Concept":      {bg:"#ec489922", fg:"#f472b6", bd:"#ec489940"},
    "Product":      {bg:"#22c55e22", fg:"#4ade80", bd:"#22c55e40"},
    "Technology":   {bg:"#3b82f622", fg:"#60a5fa", bd:"#3b82f640"},
    "Document":     {bg:"#7c3aed22", fg:"#a78bfa", bd:"#7c3aed40"},
    "Project":      {bg:"#0ea5e922", fg:"#38bdf8", bd:"#0ea5e940"},
    "Task":         {bg:"#f9731622", fg:"#fb923c", bd:"#f9731640"},
    "Role":         {bg:"#ec489922", fg:"#f9a8d4", bd:"#ec489940"},
    "Department":   {bg:"#10b98122", fg:"#34d399", bd:"#10b98140"},
    "Event":        {bg:"#a855f722", fg:"#c084fc", bd:"#a855f740"},
    "Date":         {bg:"#eab30822", fg:"#fcd34d", bd:"#eab30840"},
    "Risk":         {bg:"#ef444422", fg:"#f87171", bd:"#ef444440"},
    "Skill":        {bg:"#06b6d422", fg:"#67e8f9", bd:"#06b6d440"},
    "Policy":       {bg:"#fb718522", fg:"#fda4af", bd:"#fb718540"},
    "Decision":     {bg:"#f9731622", fg:"#fdba74", bd:"#f9731640"},
    "Regulation":   {bg:"#a855f722", fg:"#d8b4fe", bd:"#a855f740"},
    "Contract":     {bg:"#22c55e22", fg:"#86efac", bd:"#22c55e40"},
    "Resource":     {bg:"#3b82f622", fg:"#93c5fd", bd:"#3b82f640"},
    "Deliverable":  {bg:"#ef444422", fg:"#fca5a5", bd:"#ef444440"},
    "Qualification":{bg:"#06b6d422", fg:"#a5f3fc", bd:"#06b6d440"},
    "Metric":       {bg:"#84cc1622", fg:"#d9f99d", bd:"#84cc1640"},
    "Course":       {bg:"#a855f722", fg:"#e9d5ff", bd:"#a855f740"},
    "Standard":     {bg:"#3b82f622", fg:"#bfdbfe", bd:"#3b82f640"},
  };
  var defaultPill = {bg:"#6b6b7822", fg:"#9b9baa", bd:"#6b6b7840"};

  var panel = document.getElementById("nodeInfoPanel");

  function escHtml(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function showPanel(nodeId) {
    var info = meta[nodeId];
    if (!info) return;

    var p = pillColors[info.type] || defaultPill;
    var pillStyle = "background:"+p.bg+";color:"+p.fg+";border:1px solid "+p.bd+";";

    var html = '<button class="panel-close" onclick="document.getElementById(\\'nodeInfoPanel\\').style.display=\\'none\\'">&times;</button>';
    html += '<div class="node-name">' + escHtml(nodeId) + '</div>';
    html += '<span class="node-type" style="' + pillStyle + '">' + escHtml(info.type) + '</span>';

    // Description
    if (info.description) {
      html += '<div class="node-desc">' + escHtml(info.description) + '</div>';
    }
    
    // Properties (value, unit, period, etc.)
    var props = info.properties || {};
    var propKeys = Object.keys(props);
    if (propKeys.length > 0) {
      html += '<div class="section-label">Properties</div>';
      for (var i = 0; i < propKeys.length; i++) {
        var key = propKeys[i];
        var val = props[key];
        // Prettify key: "start_date" → "Start Date"
        var label = key.replace(/_/g, " ").replace(/\b\w/g, function(c){ return c.toUpperCase(); });
        html += '<div class="conn-sentence"><span style="color:#6b6b78;">' + escHtml(label) + ':</span> <span class="conn-highlight">' + escHtml(String(val)) + '</span></div>';
      }
    }

    // Connections as readable sentences
    var conns = info.connections || [];
    if (conns.length > 0) {
      html += '<div class="section-label">Relationships</div>';
      for (var i = 0; i < conns.length; i++) {
        html += '<div class="conn-sentence">' + escHtml(conns[i].sentence) + '</div>';
      }
    }

    // Source attribution
    if (info.source) {
      var sources = info.source.split(",");
      for (var i = 0; i < sources.length; i++) {
        var src = sources[i].trim();
        if (src) {
          html += '<div class="source-tag"><span class="source-icon">📄</span><span class="source-name">' + escHtml(src) + '</span></div>';
        }
      }
    }

    html += '<div class="click-hint">Click empty space to close</div>';

    panel.innerHTML = html;
    panel.style.display = "block";
  }

  var checkInterval = setInterval(function() {
    if (typeof network !== "undefined" && network) {
      clearInterval(checkInterval);
      network.on("click", function(params) {
        if (params.nodes && params.nodes.length > 0) {
          showPanel(params.nodes[0]);
        } else {
          panel.style.display = "none";
        }
      });
    }
  }, 200);
})();
</script>
"""

        # Inject before closing </body>
        html_str = html_str.replace("</body>", injection + "</body>")

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