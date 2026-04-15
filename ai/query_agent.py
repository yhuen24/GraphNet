"""
Query Agent module for GraphNet.
Handles natural language queries against the knowledge graph.

v2 — Dual-mode: generates Cypher for Neo4j, uses semantic search for embedded mode.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from config import config
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


class QueryAgent:
    """Agent for processing natural language queries against the knowledge graph."""

    def __init__(self, graph_manager):
        """
        Args:
            graph_manager: Either GraphManager (Neo4j) or EmbeddedGraphManager.
        """
        self.graph_manager = graph_manager
        self.llm = None
        self.initialized = False
        # Detect mode from the class name so we know which query path to use
        self._embedded = type(graph_manager).__name__ == "EmbeddedGraphManager"

    def initialize(self) -> bool:
        try:
            if not config.GOOGLE_API_KEY:
                logger.error("Google API key not configured")
                return False

            self.llm = ChatGoogleGenerativeAI(
                model=config.AI_MODEL,
                temperature=0,
                google_api_key=config.GOOGLE_API_KEY,
            )
            self.initialized = True
            mode = "embedded (semantic search)" if self._embedded else "Neo4j (Cypher)"
            logger.info(f"Query agent initialized — {mode}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            return False

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
    # EMBEDDED MODE — semantic search + LLM explanation
    # =====================================================================

    def _process_embedded_query(self, query: str) -> Dict[str, Any]:
        """
        Pipeline for embedded (NetworkX) graphs:
            1. Ask LLM to extract key search terms from the user query.
            2. Run semantic + fuzzy search in the graph manager.
            3. Optionally expand with neighbourhood traversal.
            4. Ask LLM to explain results in natural language.
        """
        # Step 1 — extract search intent
        search_terms = self._extract_search_terms(query)
        logger.info(f"Extracted search terms: {search_terms}")

        # Step 2 — gather results from graph
        all_results: List[Dict[str, Any]] = []
        matched_node_ids: List[str] = []

        for term in search_terms:
            # Semantic search (embeddings)
            sem_results = self.graph_manager.semantic_search(term, top_k=5)
            for r in sem_results:
                nid = r.get("node_id")
                if nid and nid not in matched_node_ids:
                    matched_node_ids.append(nid)

            # Also try the full query through query_graph (fuzzy + pattern)
            qr = self.graph_manager.query_graph(term)
            all_results.extend(qr)

        # Full-query search too (handles "show all organizations" etc.)
        qr_full = self.graph_manager.query_graph(query)
        all_results.extend(qr_full)

        # Step 3 — expand neighbourhood for matched nodes
        if matched_node_ids:
            neighbourhood = self.graph_manager.get_neighbourhood(
                matched_node_ids, depth=1
            )
            # Convert neighbourhood edges to result rows
            for edge in neighbourhood.get("edges", []):
                src_id = edge["source"]
                tgt_id = edge["target"]
                src_data = {}
                tgt_data = {}
                for n in neighbourhood["nodes"]:
                    if n["id"] == src_id:
                        src_data = n.get("properties", {})
                    if n["id"] == tgt_id:
                        tgt_data = n.get("properties", {})

                all_results.append({
                    "entity": src_data.get("name", "Unknown"),
                    "entity_type": [src_data.get("type", "Unknown")],
                    "source": src_data.get("source", ""),
                    "relationship": edge.get("type", "RELATED_TO"),
                    "connected_entity": tgt_data.get("name", "Unknown"),
                    "connected_type": [tgt_data.get("type", "Unknown")],
                    "connected_source": tgt_data.get("source", ""),
                })

        # Deduplicate
        all_results = self._deduplicate_results(all_results)

        # Extract sources
        sources = self._extract_sources(all_results)

        # Step 4 — LLM explanation
        explanation = self._explain_embedded_results(query, all_results, sources)

        return {
            "success": True,
            "query": f"[semantic search: {search_terms}]",
            "results": all_results,
            "explanation": explanation,
            "result_count": len(all_results),
            "sources": sources,
        }

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
- "Find people who work at Google" → ["Google", "people"]
- "What do you know about climate change?" → ["climate change"]
- "List all locations" → ["locations"]

Return ONLY the JSON array."""),
                HumanMessage(content=query),
            ])
            response = self.llm.invoke(prompt.format_messages())
            text = response.content.strip().strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()

            import json
            terms = json.loads(text)
            if isinstance(terms, list):
                return [t for t in terms if isinstance(t, str) and t.strip()]
        except Exception as e:
            logger.warning(f"LLM term extraction failed: {e}")

        # Fallback: use the raw query
        return [query]

    def _explain_embedded_results(self, query: str,
                                   results: List[Dict], sources: List[str]) -> str:
        """Generate a natural-language explanation of the search results."""
        if not results:
            return ("I couldn't find any matching entities in the knowledge graph. "
                    "Try rephrasing your question or uploading more documents.")
        try:
            source_info = ""
            if sources:
                source_info = f"\nSource documents: {', '.join(sources)}"

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are explaining knowledge graph search results.
Give a clear, concise answer to the user's question based on the graph data provided.
Mention specific entity names and relationships.
If source documents are listed, mention them at the end as: " Source: [name]"
Do NOT say "the search results show" — just answer the question directly."""),
                HumanMessage(content=f"""
Question: {query}

Graph data found ({len(results)} results):
{self._format_results_for_llm(results)}
{source_info}

Answer the question based on this data."""),
            ])
            response = self.llm.invoke(prompt.format_messages())
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error explaining results: {e}")
            return f"Found {len(results)} results related to your query."

    @staticmethod
    def _format_results_for_llm(results: List[Dict], max_rows: int = 20) -> str:
        """Format result rows into a compact text block for the LLM."""
        lines = []
        for r in results[:max_rows]:
            if "relationship" in r and "connected_entity" in r:
                lines.append(
                    f"- {r.get('entity', '?')} —[{r['relationship']}]→ "
                    f"{r['connected_entity']}  (source: {r.get('source', '?')})"
                )
            elif "name" in r:
                lines.append(
                    f"- {r['name']} (type: {r.get('type', '?')}, "
                    f"source: {r.get('source', '?')})"
                )
            else:
                lines.append(f"- {r}")
        if len(results) > max_rows:
            lines.append(f"  … and {len(results) - max_rows} more")
        return "\n".join(lines)

    # =====================================================================
    # NEO4J MODE — Cypher generation (kept for Neo4j users)
    # =====================================================================

    def _process_cypher_query(self, query: str) -> Dict[str, Any]:
        """
        Neo4j pipeline: Cypher + semantic search merged.
        Always runs both and combines results so that "dangote group"
        and "dangote" find the same entities.
        """
        cypher = self.generate_cypher_query(query)
        if not cypher:
            return {"success": False, "error": "Could not generate Cypher query",
                    "results": []}

        results = self.graph_manager.query_graph(cypher)

        # Fallback 1: broader Cypher search (only if Cypher returned nothing)
        if not results:
            fallback = self._generate_fallback_query(query)
            if fallback:
                results = self.graph_manager.query_graph(fallback)
                if results:
                    cypher = fallback

        # Always augment with semantic search — this ensures "dangote group"
        # and "dangote" both find the same entities regardless of how the
        # Cypher CONTAINS substring matches.
        if hasattr(self.graph_manager, "semantic_search"):
            search_terms = self._extract_search_terms(query)
            matched_names = []

            # Collect names already in Cypher results so we don't duplicate
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

            # Expand neighbourhood around new semantic matches
            if matched_names and hasattr(self.graph_manager, "get_neighbourhood"):
                neighbourhood = self.graph_manager.get_neighbourhood(
                    matched_names, depth=1
                )
                for edge in neighbourhood.get("edges", []):
                    results.append({
                        "entity": edge.get("source", "?"),
                        "relationship": edge.get("type", "RELATED_TO"),
                        "connected_entity": edge.get("target", "?"),
                    })

            # If Cypher returned nothing, include the entity info directly
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

        # Deduplicate
        results = self._deduplicate_results(results)

        sources = self._extract_sources(results)
        explanation = self.explain_results(query, cypher, results, sources)

        return {
            "success": True,
            "query": cypher,
            "results": results,
            "explanation": explanation,
            "result_count": len(results),
            "sources": sources,
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
            cypher = response.content.strip()
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
            term = response.content.strip().lower()
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

    def explain_results(self, natural_query: str, cypher_query: str,
                        results: List[Dict], sources: List[str] = None) -> str:
        if not self.initialized:
            return "Query agent not initialized"
        try:
            source_info = ""
            if sources:
                source_info = f"\nSource documents: {', '.join(sources)}"

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are explaining query results from a knowledge graph.
Provide a clear, concise explanation. Be specific and mention entity names.
Always mention source documents at the end: "📄 Source: [name]" """),
                HumanMessage(content=f"""
Original question: {natural_query}
Query executed: {cypher_query}
Results found: {len(results)}
Sample results: {str(results[:5]) if results else "No results"}
{source_info}

Provide a brief, natural explanation."""),
            ])
            response = self.llm.invoke(prompt.format_messages())
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error explaining results: {e}")
            return f"Found {len(results)} results"

    # =====================================================================
    # Shared helpers
    # =====================================================================

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
        """Remove duplicate result rows based on key fields."""
        seen = set()
        deduped = []
        for r in results:
            # Build a hashable signature
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
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error summarizing: {e}")
            return f"{entity_name} — {len(relationships)} relationships"

    def get_suggestions(self, partial_query: str) -> List[str]:
        suggestions = [
            "Show me all entities",
            "Find all organizations",
            "What are the relationships for [entity name]?",
            "Find entities related to [entity name]",
            "Show all people in the graph",
            "List all locations",
            "What does [entity name] relate to?",
            "Find connections between [entity1] and [entity2]",
        ]
        if partial_query:
            suggestions = [s for s in suggestions
                           if partial_query.lower() in s.lower()]
        return suggestions[:5]