"""
Query Agent module for GraphNet.
Handles natural language queries against the knowledge graph.

v2 — Dual-mode: generates Cypher for Neo4j, uses semantic search for embedded mode.
v3 — Added _get_text helper to support Gemini 3 Flash list-of-blocks responses.
v4 — REASONING UPGRADE: retrieves document chunks alongside graph data, uses
     analysis-focused prompts so the LLM can make decisions, judgments, and
     comparisons instead of just describing search results.
"""

import json
import logging
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from config import config
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# REASONING SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

REASONING_SYSTEM_PROMPT = """You are GraphNet's AI analyst. You have access to two types of evidence:

1. **GRAPH DATA** — structured entities and relationships extracted from uploaded documents.
2. **DOCUMENT CONTEXT** — the original text passages from those documents.

Your job is to REASON about this evidence to answer the user's question. You are NOT a search engine that just lists what was found. You are an analyst that:

- **Answers directly** — start with a clear Yes/No/answer when the question calls for one.
- **Makes judgments** — if asked whether something is in scope, compliant, or relevant, DECIDE and explain your reasoning.
- **Cites evidence** — back up your answer by referencing specific entities, relationships, or document passages.
- **Identifies gaps** — if the evidence is insufficient, say what's missing rather than guessing.
- **Connects the dots** — draw inferences across multiple pieces of evidence when needed.

RESPONSE STRUCTURE:
1. Lead with your answer/decision (1-2 sentences).
2. Explain your reasoning with specific evidence from the graph and documents.
3. If relevant, note any caveats or limitations.
4. End with source attribution: "📄 Source: [document name]"

RULES:
- NEVER say "the search results show" or "based on the graph data" — just answer naturally.
- If the question is about a policy, regulation, or standard — analyze whether the conditions are met.
- If the question is comparative — provide a structured comparison.
- If the question is exploratory — give a comprehensive but concise overview.
- If you genuinely cannot answer from the available evidence, say so clearly and suggest what documents might help.
- When entities and relationships are relevant to your answer, mention them by name so the graph visualization can highlight them."""


REASONING_USER_TEMPLATE = """Question: {query}

═══ GRAPH DATA ({graph_count} results) ═══
{graph_data}

═══ DOCUMENT CONTEXT ({chunk_count} relevant passages) ═══
{document_context}

═══ SOURCE DOCUMENTS ═══
{sources}

Analyze this evidence and answer the question. Remember: decide, don't just describe."""


# ═════════════════════════════════════════════════════════════════════════════

class QueryAgent:
    """Agent for processing natural language queries against the knowledge graph."""

    def __init__(self, graph_manager, chunk_store=None):
        """
        Args:
            graph_manager: Either GraphManager (Neo4j) or EmbeddedGraphManager.
            chunk_store:   Optional ChunkStore for document context retrieval.
        """
        self.graph_manager = graph_manager
        self.chunk_store = chunk_store
        self.llm = None
        self.initialized = False
        self._embedded = type(graph_manager).__name__ == "EmbeddedGraphManager"

    def initialize(self) -> bool:
        try:
            if not config.GOOGLE_API_KEY:
                logger.error("Google API key not configured")
                return False

            self.llm = ChatGoogleGenerativeAI(
                model=config.AI_MODEL,
                temperature=0.1,  # Slight creativity for reasoning
                google_api_key=config.GOOGLE_API_KEY,
            )
            self.initialized = True
            mode = "embedded (semantic search)" if self._embedded else "Neo4j (Cypher)"
            has_chunks = "with document context" if self.chunk_store else "without document context"
            logger.info(f"Query agent initialized — {mode}, {has_chunks}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            return False

    # =====================================================================
    # Response text helper (Gemini 3 Flash compatibility)
    # =====================================================================

    @staticmethod
    def _get_text(response) -> str:
        """
        Extract plain text from an LLM response.
        Handles both string content and list-of-blocks content.
        """
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text" or "text" in block:
                        parts.append(block.get("text", ""))
            return "".join(parts)
        return str(content)

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        if not self.initialized:
            return {"success": False, "error": "Query agent not initialized", "results": []}

        try:
            if self._embedded:
                return self._process_embedded_query(natural_language_query)
            else:
                return self._process_cypher_query(natural_language_query)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"success": False, "error": str(e), "results": []}

    # =====================================================================
    # EMBEDDED MODE — semantic search + document context + reasoning
    # =====================================================================

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        """Case-insensitive similarity ratio between two strings."""
        return SequenceMatcher(
            None, a.strip().lower(), b.strip().lower()
        ).ratio()

    def _process_embedded_query(self, query: str) -> Dict[str, Any]:
        """
        Upgraded pipeline for embedded (NetworkX) graphs:
            1. Extract search terms from the query.
            2. Find focal entities via semantic + fuzzy search.
            3. Expand 1-hop neighbourhood around focal entities.
            4. Retrieve relevant document chunks (NEW).
            5. Send graph data + document context to reasoning LLM (NEW).
        """
        # Step 1 — extract search intent
        search_terms = self._extract_search_terms(query)
        logger.info(f"Extracted search terms: {search_terms}")

        # Step 2 — identify focal entities via semantic + fuzzy search
        all_results: List[Dict[str, Any]] = []
        focal_node_ids: List[str] = []
        focal_names: set = set()
        focal_display_names: List[str] = []

        for term in search_terms:
            sem_results = self.graph_manager.semantic_search(term, top_k=10)
            for r in sem_results:
                nid = r.get("node_id")
                name = r.get("name", "")
                sim = self._name_similarity(term, name)
                if sim >= 0.45 and nid and nid not in focal_node_ids:
                    focal_node_ids.append(nid)
                    focal_names.add(name.strip().lower())
                    focal_display_names.append(name.strip())
                    logger.info(
                        f"Focal match: '{name}' (sim={sim:.2f}) for term '{term}'"
                    )

        # Step 3 — expand neighbourhood ONLY for focal entities (1-hop)
        focal_node_id_set = set(focal_node_ids)
        neighbour_names: set = set()

        if focal_node_ids:
            neighbourhood = self.graph_manager.get_neighbourhood(
                focal_node_ids, depth=1
            )
            for edge in neighbourhood.get("edges", []):
                src_id = edge["source"]
                tgt_id = edge["target"]

                if src_id not in focal_node_id_set and tgt_id not in focal_node_id_set:
                    continue

                src_data = {}
                tgt_data = {}
                src_type = "Unknown"
                tgt_type = "Unknown"
                for n in neighbourhood["nodes"]:
                    if n["id"] == src_id:
                        src_data = n.get("properties", {})
                        src_type = n.get("type") or src_data.get("type", "Unknown")
                    if n["id"] == tgt_id:
                        tgt_data = n.get("properties", {})
                        tgt_type = n.get("type") or tgt_data.get("type", "Unknown")

                src_name = src_data.get("name", "Unknown")
                tgt_name = tgt_data.get("name", "Unknown")

                neighbour_names.add(src_name.strip().lower())
                neighbour_names.add(tgt_name.strip().lower())

                all_results.append({
                    "entity": src_name,
                    "entity_type": [src_type],
                    "source": src_data.get("source", ""),
                    "relationship": edge.get("type", "RELATED_TO"),
                    "connected_entity": tgt_name,
                    "connected_type": [tgt_type],
                    "connected_source": tgt_data.get("source", ""),
                })

        # Step 4 — filter query_graph results to focal/neighbour only
        allowed_names = focal_names | neighbour_names

        for term in search_terms:
            qr = self.graph_manager.query_graph(term)
            for row in qr:
                if self._row_involves_allowed(row, allowed_names):
                    all_results.append(row)

        qr_full = self.graph_manager.query_graph(query)
        for row in qr_full:
            if self._row_involves_allowed(row, allowed_names):
                all_results.append(row)

        all_results = self._deduplicate_results(all_results)
        sources = self._extract_sources(all_results)

        # ──────────────────────────────────────────────────────────────────
        # Step 5 — NEW: retrieve document chunks for reasoning context
        # ──────────────────────────────────────────────────────────────────
        document_chunks = self._retrieve_document_context(query, search_terms, sources)

        # Step 6 — NEW: reasoning-based explanation
        explanation = self._reason_over_evidence(
            query, all_results, document_chunks, sources
        )

        # Derive focal entities from the LLM's actual answer
        answer_focal = []
        if explanation:
            explanation_lower = explanation.lower()
            candidate_names = set()
            # Names from structured results
            for row in all_results:
                for key in ("entity", "name", "connected_entity"):
                    val = row.get(key)
                    if isinstance(val, str) and val.strip():
                        candidate_names.add(val.strip())
            # Also include names found by semantic search — these may
            # not appear in all_results but could be in the LLM answer
            for name in focal_display_names:
                candidate_names.add(name)
            for name in candidate_names:
                if name.lower() in explanation_lower and name not in answer_focal:
                    answer_focal.append(name)

        # Fallback: if LLM used different phrasing and nothing matched,
        # keep the search-derived focal names (capped)
        if not answer_focal:
            answer_focal = focal_display_names[:5]

        return {
            "success": True,
            "query": f"[semantic search: {search_terms}]",
            "results": all_results,
            "explanation": explanation,
            "result_count": len(all_results),
            "sources": sources,
            "focal_entities": answer_focal,
        }

    # =====================================================================
    # DOCUMENT CONTEXT RETRIEVAL (NEW)
    # =====================================================================

    def _retrieve_document_context(self, query: str,
                                    search_terms: List[str],
                                    sources: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks to give the LLM actual text
        to reason about, not just entity names.

        Strategy:
        1. Semantic search over chunks using the full query.
        2. Also search for each extracted search term.
        3. Deduplicate and return top chunks.
        """
        if not self.chunk_store:
            return []

        seen_ids = set()
        all_chunks = []

        # Search with the full query
        try:
            query_chunks = self.chunk_store.search(query, top_k=3)
            for chunk in query_chunks:
                cid = f"{chunk['source']}::{chunk['chunk_index']}"
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Chunk search failed for query: {e}")

        # Search with individual terms
        for term in search_terms:
            try:
                term_chunks = self.chunk_store.search(term, top_k=2)
                for chunk in term_chunks:
                    cid = f"{chunk['source']}::{chunk['chunk_index']}"
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Chunk search failed for term '{term}': {e}")

        # Sort by relevance score, take top N
        all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        top_chunks = all_chunks[:5]  # Max 5 chunks to avoid context overflow

        if top_chunks:
            logger.info(
                f"Retrieved {len(top_chunks)} document chunks for reasoning "
                f"(from {len(set(c['source'] for c in top_chunks))} documents)"
            )

        return top_chunks

    # =====================================================================
    # REASONING ENGINE (NEW — replaces old _explain_embedded_results)
    # =====================================================================

    def _reason_over_evidence(self, query: str,
                               graph_results: List[Dict],
                               document_chunks: List[Dict],
                               sources: List[str]) -> str:
        """
        Send graph data + document context to the LLM with a reasoning
        prompt that instructs it to analyze, judge, and decide.

        This replaces the old _explain_embedded_results which just described
        what was found.
        """
        if not graph_results and not document_chunks:
            return ("I couldn't find any matching information in the knowledge graph "
                    "or uploaded documents. Try rephrasing your question or uploading "
                    "more documents that might contain the answer.")

        try:
            # Format graph data for the LLM
            graph_text = self._format_results_for_llm(graph_results)
            if not graph_text.strip():
                graph_text = "(No structured graph data found for this query)"

            # Format document chunks — truncate each to avoid context overflow
            chunk_texts = []
            for i, chunk in enumerate(document_chunks):
                text = chunk.get("text", "")
                # Truncate very long chunks but keep enough for reasoning
                if len(text) > 3000:
                    text = text[:3000] + "… [truncated]"
                chunk_texts.append(
                    f"--- Passage {i+1} (from: {chunk.get('source', '?')}, "
                    f"relevance: {chunk.get('score', 0):.2f}) ---\n{text}"
                )
            doc_context = "\n\n".join(chunk_texts) if chunk_texts else "(No document text available)"

            source_info = ", ".join(sources) if sources else "No source documents identified"

            # Build the reasoning prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=REASONING_SYSTEM_PROMPT),
                HumanMessage(content=REASONING_USER_TEMPLATE.format(
                    query=query,
                    graph_count=len(graph_results),
                    graph_data=graph_text,
                    chunk_count=len(document_chunks),
                    document_context=doc_context,
                    sources=source_info,
                )),
            ])

            response = self.llm.invoke(prompt.format_messages())
            return self._get_text(response).strip()

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            # Fallback to basic explanation
            return self._fallback_explain(query, graph_results, sources)

    def _fallback_explain(self, query: str, results: List[Dict],
                           sources: List[str]) -> str:
        """Fallback if the reasoning prompt fails — simpler description."""
        if not results:
            return "No results found for your query."
        try:
            source_info = f"\nSource documents: {', '.join(sources)}" if sources else ""
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Answer the user's question based on the graph data.
Be specific and mention entity names. End with source attribution."""),
                HumanMessage(content=f"""
Question: {query}
Graph data ({len(results)} results):
{self._format_results_for_llm(results)}
{source_info}
Answer directly."""),
            ])
            response = self.llm.invoke(prompt.format_messages())
            return self._get_text(response).strip()
        except Exception as e:
            logger.error(f"Fallback explain failed: {e}")
            return f"Found {len(results)} results related to your query."

    # =====================================================================
    # SHARED HELPERS (unchanged from v3)
    # =====================================================================

    @staticmethod
    def _row_involves_allowed(row: Dict[str, Any],
                              allowed_names: set) -> bool:
        for key in ("entity", "name", "connected_entity",
                    "n.name", "m.name"):
            val = row.get(key)
            if isinstance(val, str) and val.strip().lower() in allowed_names:
                return True
            if isinstance(val, dict):
                n = val.get("name", "")
                if n and n.strip().lower() in allowed_names:
                    return True
        return False

    def _extract_search_terms(self, query: str) -> List[str]:
        """Use LLM to pull out the key entity names / topics from the query."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Extract the key entity names or search terms from the user's question.
Return ONLY a JSON array of strings, nothing else.

Examples:
- "Tell me about Dangote Group" → ["Dangote Group"]
- "What is the relationship between Apple and Samsung?" → ["Apple", "Samsung"]
- "Show me all organizations" → ["organizations"]
- "Is this file in scope of the data retention policy?" → ["data retention policy"]
- "Does the code of conduct cover anti-bribery?" → ["code of conduct", "anti-bribery"]
- "Compare Policy A with Policy B" → ["Policy A", "Policy B"]

Return ONLY the JSON array."""),
                HumanMessage(content=query),
            ])
            response = self.llm.invoke(prompt.format_messages())
            text = self._get_text(response).strip().strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()

            terms = json.loads(text)
            if isinstance(terms, list):
                return [t for t in terms if isinstance(t, str) and t.strip()]
        except Exception as e:
            logger.warning(f"LLM term extraction failed: {e}")

        return [query]

    @staticmethod
    def _format_results_for_llm(results: List[Dict], max_rows: int = 25) -> str:
        """Format result rows into a compact text block for the LLM."""
        lines = []
        for r in results[:max_rows]:
            if "relationship" in r and "connected_entity" in r:
                lines.append(
                    f"- {r.get('entity', '?')} —[{r['relationship']}]→ "
                    f"{r['connected_entity']}  (source: {r.get('source', '?')})"
                )
            elif "name" in r:
                desc = r.get("description", "")
                desc_part = f" — {desc[:100]}" if desc else ""
                lines.append(
                    f"- {r['name']} (type: {r.get('type', '?')}, "
                    f"source: {r.get('source', '?')}){desc_part}"
                )
            else:
                lines.append(f"- {r}")
        if len(results) > max_rows:
            lines.append(f"  … and {len(results) - max_rows} more")
        return "\n".join(lines)

    @staticmethod
    def _extract_sources(results: List[Dict]) -> List[str]:
        sources = set()
        for record in results:
            for key, value in record.items():
                if "source" in key.lower() and isinstance(value, str) and value:
                    for s in value.split(","):
                        s = s.strip()
                        if s:
                            sources.add(s)
                elif isinstance(value, dict) and value.get("source"):
                    sources.add(value["source"])
        return sorted(sources)

    @staticmethod
    def _deduplicate_results(results: List[Dict]) -> List[Dict]:
        seen = set()
        deduped = []
        for r in results:
            sig_parts = []
            for k in sorted(r.keys()):
                v = r[k]
                if isinstance(v, list):
                    v = tuple(v)
                sig_parts.append((k, v))
            sig = tuple(sig_parts)
            if sig not in seen:
                seen.add(sig)
                deduped.append(r)
        return deduped

    # =====================================================================
    # NEO4J MODE — Cypher generation + reasoning
    # =====================================================================

    def _process_cypher_query(self, query: str) -> Dict[str, Any]:
        """
        Neo4j pipeline: Cypher + semantic search + document context + reasoning.
        """
        cypher = self.generate_cypher_query(query)
        if not cypher:
            return {"success": False, "error": "Could not generate Cypher query",
                    "results": []}

        results = self.graph_manager.query_graph(cypher)

        # Fallback 1: broader Cypher search
        if not results:
            fallback = self._generate_fallback_query(query)
            if fallback:
                results = self.graph_manager.query_graph(fallback)
                if results:
                    cypher = fallback

        # Always augment with semantic search
        search_terms: List[str] = []
        if hasattr(self.graph_manager, "semantic_search"):
            search_terms = self._extract_search_terms(query)
            matched_names = []

            existing_names = set()
            for r in results:
                for key in ("entity", "name", "connected_entity"):
                    val = r.get(key)
                    if isinstance(val, str) and val:
                        existing_names.add(val.lower())

            for term in search_terms:
                hits = self.graph_manager.semantic_search(term, top_k=5)
                for hit in hits:
                    name = hit.get("name", "")
                    if name and name.lower() not in existing_names:
                        matched_names.append(name)
                        existing_names.add(name.lower())

            if matched_names and hasattr(self.graph_manager, "get_neighbourhood"):
                neighbourhood = self.graph_manager.get_neighbourhood(
                    matched_names, depth=1
                )
                _nb_nodes = {}
                for n in neighbourhood.get("nodes", []):
                    _nb_nodes[n.get("label") or n.get("id", "")] = n

                for edge in neighbourhood.get("edges", []):
                    src_name = edge.get("source", "?")
                    tgt_name = edge.get("target", "?")
                    src_node = _nb_nodes.get(src_name, {})
                    tgt_node = _nb_nodes.get(tgt_name, {})

                    results.append({
                        "entity": src_name,
                        "entity_type": [src_node.get("type")
                                        or src_node.get("properties", {}).get("type", "Unknown")],
                        "relationship": edge.get("type", "RELATED_TO"),
                        "connected_entity": tgt_name,
                        "connected_type": [tgt_node.get("type")
                                           or tgt_node.get("properties", {}).get("type", "Unknown")],
                    })

            if not results:
                for term in search_terms:
                    hits = self.graph_manager.semantic_search(term, top_k=5)
                    for hit in hits:
                        results.append({
                            "name": hit.get("name"),
                            "type": (hit.get("labels", [None]) or [None])[0]
                                    if isinstance(hit.get("labels"), list)
                                    else str(hit.get("type", "Unknown")),
                            "source": hit.get("source", ""),
                            "description": hit.get("description", ""),
                        })

        results = self._deduplicate_results(results)
        sources = self._extract_sources(results)

        # ── NEW: retrieve document chunks for reasoning ──
        document_chunks = self._retrieve_document_context(query, search_terms, sources)

        # ── NEW: reasoning-based explanation ──
        explanation = self._reason_over_evidence(query, results, document_chunks, sources)

        # Focal entities for graph rendering — derived from the LLM answer
        # so the graph only shows nodes the text response discusses.
        cypher_focal = []
        if explanation:
            explanation_lower = explanation.lower()
            candidate_names = set()
            for row in results:
                for key in ("entity", "name", "connected_entity"):
                    val = row.get(key)
                    if isinstance(val, str) and val.strip():
                        candidate_names.add(val.strip())
            # Also include semantic search hits
            if hasattr(self.graph_manager, "semantic_search"):
                for term in search_terms:
                    for hit in self.graph_manager.semantic_search(term, top_k=3):
                        name = hit.get("name", "")
                        if name:
                            candidate_names.add(name.strip())
            for name in candidate_names:
                if name.lower() in explanation_lower and name not in cypher_focal:
                    cypher_focal.append(name)

        # Fallback: if nothing matched, use semantic search hits
        if not cypher_focal and hasattr(self.graph_manager, "semantic_search"):
            _terms = search_terms or self._extract_search_terms(query)
            for term in _terms[:2]:
                hits = self.graph_manager.semantic_search(term, top_k=3)
                for hit in hits:
                    name = hit.get("name", "")
                    sim = self._name_similarity(term, name)
                    if sim >= 0.6 and name not in cypher_focal:
                        cypher_focal.append(name)

        return {
            "success": True,
            "query": cypher,
            "results": results,
            "explanation": explanation,
            "result_count": len(results),
            "sources": sources,
            "focal_entities": cypher_focal,
        }

    def generate_cypher_query(self, natural_language_query: str) -> Optional[str]:
        if not self.initialized:
            return None
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert at converting natural language questions 
into Neo4j Cypher queries. Generate ONLY the Cypher query, no explanations.

CRITICAL RULES:
1. ALWAYS use case-insensitive matching.  Nodes have a `canonical_name` property (lowercase, no articles).
   - PREFERRED:  WHERE n.canonical_name CONTAINS "dangote"   (user term must be lowercased by you)
   - FALLBACK:   WHERE toLower(n.name) CONTAINS "dangote"
2. When using CONTAINS, use the SHORTEST distinctive keyword, NOT the full phrase.
   - "Dangote Group" → CONTAINS "dangote"  (NOT "dangote group")
   - "Apple Inc" → CONTAINS "apple"  (NOT "apple inc")
   - "Bank of Nigeria" → CONTAINS "nigeria" or CONTAINS "bank"
   This ensures you catch related entities whose names partially overlap.
3. Search ALL nodes regardless of label: MATCH (n) WHERE …
4. Always include LIMIT.
5. Always return n.source for traceability.
6. For "tell me about X" or "information about X":
   MATCH (n)-[r]-(m)
   WHERE n.canonical_name CONTAINS "<shortest keyword lowercased>"
   RETURN n.name AS entity, n.source AS source, type(r) AS relationship,
          m.name AS connected_entity, m.source AS connected_source,
          labels(n) AS entity_type, labels(m) AS connected_type
   LIMIT 25
7. For "show all/list all Type": MATCH (n:Type) RETURN n.name AS name, n.source AS source LIMIT 25

Entity types: Person, Organization, Location, Concept, Product, Date, Event, Technology
Relationship types: WORKS_FOR, LOCATED_IN, RELATED_TO, OWNS, CREATED, MANAGES, PARTICIPATED_IN, PRODUCES, OPERATES_IN, SUBSIDIARY_OF, FOUNDED, HAS, IS_A, PART_OF, SUPPLIES, INVESTED_IN

Generate only the Cypher query without markdown or explanations."""),
                HumanMessage(content=f"Convert this question to Cypher: {natural_language_query}"),
            ])
            response = self.llm.invoke(prompt.format_messages())
            cypher = self._get_text(response).strip()
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            logger.info(f"Generated Cypher: {cypher}")
            return cypher
        except Exception as e:
            logger.error(f"Error generating Cypher: {str(e)}")
            return None

    def _generate_fallback_query(self, query: str) -> Optional[str]:
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Extract the main search term from this question.
Return ONLY the term, no quotes, no explanation.
If the question is about listing all of a type, return NONE."""),
                HumanMessage(content=query),
            ])
            response = self.llm.invoke(prompt.format_messages())
            term = self._get_text(response).strip().lower()
            if term and term != "none":
                return f"""
                MATCH (n)-[r]-(m)
                WHERE toLower(n.name) CONTAINS "{term}"
                RETURN n.name AS entity, n.source AS source, labels(n) AS entity_type,
                       type(r) AS relationship,
                       m.name AS connected_entity, m.source AS connected_source, labels(m) AS connected_type
                LIMIT 25
                """
        except Exception as e:
            logger.error(f"Fallback query error: {e}")
        return None

    # =====================================================================
    # Entity info & suggestions
    # =====================================================================

    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        try:
            entity = self.graph_manager.get_entity(entity_name)
            relationships = self.graph_manager.get_entity_relationships(entity_name)
            summary = (self.summarize_entity(entity_name, entity, relationships)
                       if self.initialized and entity
                       else f"Entity: {entity_name}")
            return {
                "success": True,
                "entity": entity,
                "relationships": relationships,
                "summary": summary,
            }
        except Exception as e:
            logger.error(f"Error getting entity info: {str(e)}")
            return {"success": False, "error": str(e)}

    def summarize_entity(self, entity_name: str, entity_data: Dict,
                         relationships: List[Dict]) -> str:
        try:
            source = entity_data.get("source", "Unknown")
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Summarize this entity from a knowledge graph.
                Mention properties, key relationships, and the source document."""),
                                HumanMessage(content=f"""
                Entity: {entity_name}
                Properties: {entity_data}
                Relationships: {relationships}
                Source: {source}"""),
                            ])
            response = self.llm.invoke(prompt.format_messages())
            return self._get_text(response).strip()
        except Exception as e:
            logger.error(f"Error summarizing: {e}")
            return f"{entity_name} — {len(relationships)} relationships"

