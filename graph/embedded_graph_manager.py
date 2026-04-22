"""
Embedded Graph Manager module for GraphNet.
Provides in-memory graph storage using NetworkX (no Neo4j required).

v2 — Canonical keying, fuzzy search, embedding integration.
"""

import logging
import pickle
from typing import List, Dict, Any, Optional, Set
import networkx as nx
from datetime import datetime

from utils.normalization import canonical_key, display_name, fuzzy_match, find_best_match

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddedGraphManager:
    """
    Manages in-memory graph using NetworkX.
    Alternative to Neo4j — no database installation required!

    Node IDs are canonical:  "<lowercase_type>:<canonical_name>"
    This ensures "Company A", "company a", "COMPANY A" all map to one node.
    """

    def __init__(self, persist_file: str = "graphnet_data.pkl"):
        self.graph = nx.MultiDiGraph()
        self.persist_file = persist_file
        self.connected = False
        self.embedding_store = None  # set externally after init

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            try:
                with open(self.persist_file, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded existing graph from {self.persist_file}")
            except FileNotFoundError:
                logger.info("Creating new graph (no existing data found)")
                self.graph = nx.MultiDiGraph()

            self.connected = True
            logger.info("✓ Embedded graph initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedded graph: {str(e)}")
            self.connected = False
            return False

    def close(self):
        if self.connected:
            self._persist()
            if self.embedding_store:
                self.embedding_store.save()
            self.connected = False
            logger.info("Embedded graph saved and closed")

    def _persist(self):
        try:
            with open(self.persist_file, "wb") as f:
                pickle.dump(self.graph, f)
        except Exception as e:
            logger.error(f"Error persisting graph: {str(e)}")

    # ── Canonical ID helpers ─────────────────────────────────────────────────

    @staticmethod
    def _make_node_id(entity_type: str, entity_name: str) -> str:
        """
        Build a canonical node ID:  "organization:dangote group"
        The type is lowercased, the name is canonical-keyed.
        """
        ctype = entity_type.strip().lower() if entity_type else "entity"
        cname = canonical_key(entity_name)
        return f"{ctype}:{cname}"

    def _resolve_node_id(self, entity_name: str,
                         entity_type: str = None) -> Optional[str]:
        """
        Find the node ID for *entity_name*, trying (in order):
            1. Exact canonical match  (with type if given)
            2. Exact canonical match  (any type)
            3. Fuzzy match across all node display names
        """
        cname = canonical_key(entity_name)

        # 1. Exact with type
        if entity_type:
            nid = self._make_node_id(entity_type, entity_name)
            if self.graph.has_node(nid):
                return nid

        # 2. Exact without type — scan all nodes
        for nid in self.graph.nodes():
            node_cname = canonical_key(self.graph.nodes[nid].get("name", ""))
            if node_cname == cname:
                return nid

        # 3. Fuzzy fallback
        all_names = [self.graph.nodes[n].get("name", "") for n in self.graph.nodes()]
        best = find_best_match(entity_name, all_names, threshold=0.60)
        if best:
            for nid in self.graph.nodes():
                if self.graph.nodes[nid].get("name", "") == best:
                    return nid

        return None

    # ── CRUD operations ──────────────────────────────────────────────────────

    def create_entity(self, entity_name: str, entity_type: str,
                      properties: Dict[str, Any] = None,
                      source: str = None) -> bool:
        if not self.connected:
            return False
        try:
            clean = display_name(entity_name)
            node_id = self._make_node_id(entity_type, entity_name)

            attrs = properties or {}
            attrs["name"] = clean
            attrs["type"] = entity_type
            attrs["source"] = source

            if self.graph.has_node(node_id):
                existing = self.graph.nodes[node_id]
                # Merge descriptions if new one adds info
                old_desc = existing.get("description", "")
                new_desc = attrs.get("description", "")
                if new_desc and new_desc != old_desc:
                    attrs["description"] = (
                        f"{old_desc} | {new_desc}" if old_desc else new_desc
                    )
                # Merge sources
                old_src = existing.get("source", "")
                if source and source not in (old_src or ""):
                    attrs["source"] = f"{old_src}, {source}" if old_src else source
                self.graph.nodes[node_id].update(attrs)
                self.graph.nodes[node_id]["updated"] = datetime.now().isoformat()
            else:
                attrs["created"] = datetime.now().isoformat()
                self.graph.add_node(node_id, **attrs)

            # Index in embedding store
            if self.embedding_store:
                self.embedding_store.add_entity(
                    node_id=node_id,
                    name=clean,
                    entity_type=entity_type,
                    description=attrs.get("description", ""),
                )

            self._persist()
            return True
        except Exception as e:
            logger.error(f"Error creating entity: {str(e)}")
            return False

    def create_relationship(self, source_entity: str, source_type: str,
                            target_entity: str, target_type: str,
                            relationship_type: str,
                            properties: Dict[str, Any] = None) -> bool:
        if not self.connected:
            return False
        try:
            source_id = self._resolve_node_id(source_entity, source_type)
            target_id = self._resolve_node_id(target_entity, target_type)

            # Auto-create nodes if not found
            if not source_id:
                self.create_entity(source_entity, source_type or "Entity")
                source_id = self._make_node_id(source_type or "Entity", source_entity)
            if not target_id:
                self.create_entity(target_entity, target_type or "Entity")
                target_id = self._make_node_id(target_type or "Entity", target_entity)

            attrs = properties or {}
            attrs["type"] = relationship_type
            attrs["created"] = datetime.now().isoformat()

            self.graph.add_edge(source_id, target_id, **attrs)
            self._persist()
            return True
        except Exception as e:
            logger.error(f"Error creating relationship: {str(e)}")
            return False

    # ── Read operations (all use fuzzy resolution) ───────────────────────────

    def get_entity(self, entity_name: str,
                   entity_type: str = None) -> Optional[Dict[str, Any]]:
        if not self.connected:
            return None
        nid = self._resolve_node_id(entity_name, entity_type)
        if nid:
            return dict(self.graph.nodes[nid])
        return None

    def get_entity_relationships(self, entity_name: str,
                                 entity_type: str = None) -> List[Dict[str, Any]]:
        if not self.connected:
            return []
        try:
            nid = self._resolve_node_id(entity_name, entity_type)
            if not nid:
                return []

            relationships = []
            for target in self.graph.successors(nid):
                for edge_data in self.graph.get_edge_data(nid, target).values():
                    relationships.append({
                        "relationship": edge_data.get("type", "RELATED_TO"),
                        "entity": self.graph.nodes[target].get("name", "Unknown"),
                        "labels": [self.graph.nodes[target].get("type", "Unknown")],
                        "direction": "outgoing",
                        "source": self.graph.nodes[target].get("source", ""),
                    })
            for source in self.graph.predecessors(nid):
                for edge_data in self.graph.get_edge_data(source, nid).values():
                    relationships.append({
                        "relationship": edge_data.get("type", "RELATED_TO"),
                        "entity": self.graph.nodes[source].get("name", "Unknown"),
                        "labels": [self.graph.nodes[source].get("type", "Unknown")],
                        "direction": "incoming",
                        "source": self.graph.nodes[source].get("source", ""),
                    })
            return relationships
        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}")
            return []

    # ── Search operations ────────────────────────────────────────────────────

    def search_entities(self, search_term: str,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """Fuzzy search over all entity names."""
        if not self.connected:
            return []
        try:
            all_names = []
            name_to_nid = {}
            for nid in self.graph.nodes():
                name = self.graph.nodes[nid].get("name", "")
                all_names.append(name)
                name_to_nid[name] = nid

            matches = fuzzy_match(search_term, all_names,
                                  threshold=0.45, top_k=limit)
            results = []
            for name, score in matches:
                nid = name_to_nid[name]
                data = self.graph.nodes[nid]
                results.append({
                    "name": name,
                    "types": [data.get("type", "Unknown")],
                    "score": round(score, 3),
                    "source": data.get("source", ""),
                })
            return results
        except Exception as e:
            logger.error(f"Error searching entities: {str(e)}")
            return []

    def semantic_search(self, query: str,
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Use the embedding store for deeper semantic matching.
        Returns enriched entity dicts with relationships included.
        """
        if not self.embedding_store:
            # Fall back to fuzzy string search
            return self.search_entities(query, limit=top_k)

        hits = self.embedding_store.search(query, top_k=top_k, threshold=0.20)
        results = []
        for node_id, score in hits:
            if not self.graph.has_node(node_id):
                continue
            data = dict(self.graph.nodes[node_id])
            data["node_id"] = node_id
            data["score"] = round(score, 3)
            results.append(data)
        return results

    def get_neighbourhood(self, node_ids: List[str],
                          depth: int = 1) -> Dict[str, Any]:
        """
        Return the subgraph around the given node IDs up to *depth* hops.
        Used by the query agent to pull context for the LLM.
        """
        if not self.connected:
            return {"nodes": [], "edges": []}

        visited: Set[str] = set()
        frontier = set(nid for nid in node_ids if self.graph.has_node(nid))

        for _ in range(depth):
            next_frontier: Set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                next_frontier.update(self.graph.successors(nid))
                next_frontier.update(self.graph.predecessors(nid))
            frontier = next_frontier - visited

        visited.update(frontier)

        nodes = []
        for nid in visited:
            if not self.graph.has_node(nid):
                continue
            d = self.graph.nodes[nid]
            nodes.append({
                "id": nid,
                "label": d.get("name", "Unknown"),
                "type": d.get("type", "Unknown"),
                "properties": dict(d),
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in visited and v in visited:
                edges.append({
                    "source": u,
                    "target": v,
                    "type": data.get("type", "RELATED_TO"),
                    "properties": dict(data),
                })

        return {"nodes": nodes, "edges": edges}

    # ── Intelligent query (replaces the old stub) ────────────────────────────

    def query_graph(self, query: str,
                    parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Smart query that combines fuzzy + semantic search.
        Called by the QueryAgent; replaces the old keyword-only stub.
        """
        if not self.connected:
            return []

        try:
            query_lower = query.lower().strip()

            # ── Pattern: "count" ─────────────────────────────────────────
            if "count" in query_lower:
                return [{"count": self.graph.number_of_nodes()}]

            # ── Pattern: "all" / "list" / "show" → dump by type ─────────
            type_keywords = {
                "person": "Person", "people": "Person",
                "organization": "Organization", "company": "Organization",
                "location": "Location", "place": "Location",
                "concept": "Concept", "technology": "Technology",
                "product": "Product", "event": "Event",
            }

            for kw, etype in type_keywords.items():
                if kw in query_lower and ("all" in query_lower or
                                          "list" in query_lower or
                                          "show" in query_lower or
                                          "find" in query_lower):
                    results = []
                    for nid in self.graph.nodes():
                        nd = self.graph.nodes[nid]
                        if nd.get("type", "").lower() == etype.lower():
                            results.append({
                                "name": nd.get("name"),
                                "type": nd.get("type"),
                                "source": nd.get("source", ""),
                                "description": nd.get("description", ""),
                            })
                    return results[:50]

            # ── General: semantic + fuzzy search → neighbourhood ─────────
            # 1. Collect candidate entities
            candidates = []

            # Semantic hits
            sem_hits = self.semantic_search(query, top_k=5)
            for hit in sem_hits:
                nid = hit.get("node_id")
                if nid:
                    candidates.append(nid)

            # Fuzzy string hits
            fuzz_hits = self.search_entities(query, limit=5)
            for hit in fuzz_hits:
                nid = self._resolve_node_id(hit["name"])
                if nid and nid not in candidates:
                    candidates.append(nid)

            if not candidates:
                # Last resort: dump a few nodes
                return [{"n": dict(self.graph.nodes[nid])}
                        for nid in list(self.graph.nodes())[:20]]

            # 2. Gather their relationships
            results = []
            seen_pairs = set()
            for nid in candidates:
                nd = self.graph.nodes[nid]
                entity_name = nd.get("name", "Unknown")
                entity_type = nd.get("type", "Unknown")
                entity_source = nd.get("source", "")

                # Outgoing
                for target in self.graph.successors(nid):
                    td = self.graph.nodes[target]
                    for edge_data in self.graph.get_edge_data(nid, target).values():
                        pair_key = (nid, target, edge_data.get("type"))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)
                        results.append({
                            "entity": entity_name,
                            "entity_type": [entity_type],
                            "source": entity_source,
                            "relationship": edge_data.get("type", "RELATED_TO"),
                            "connected_entity": td.get("name", "Unknown"),
                            "connected_type": [td.get("type", "Unknown")],
                            "connected_source": td.get("source", ""),
                        })

                # Incoming
                for source in self.graph.predecessors(nid):
                    sd = self.graph.nodes[source]
                    for edge_data in self.graph.get_edge_data(source, nid).values():
                        pair_key = (source, nid, edge_data.get("type"))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)
                        results.append({
                            "entity": entity_name,
                            "entity_type": [entity_type],
                            "source": entity_source,
                            "relationship": edge_data.get("type", "RELATED_TO"),
                            "connected_entity": sd.get("name", "Unknown"),
                            "connected_type": [sd.get("type", "Unknown")],
                            "connected_source": sd.get("source", ""),
                        })

            # Also include entity info even if no relationships
            if not results:
                for nid in candidates:
                    nd = self.graph.nodes[nid]
                    results.append({
                        "name": nd.get("name"),
                        "type": nd.get("type"),
                        "source": nd.get("source", ""),
                        "description": nd.get("description", ""),
                    })

            return results[:50]

        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []

    # ── Stats / graph data / clear ───────────────────────────────────────────

    def get_graph_stats(self) -> Dict[str, Any]:
        if not self.connected:
            return {}
        try:
            type_counts: Dict[str, int] = {}
            for nid in self.graph.nodes():
                nt = self.graph.nodes[nid].get("type", "Unknown")
                type_counts[nt] = type_counts.get(nt, 0) + 1

            rel_counts: Dict[str, int] = {}
            for _, _, data in self.graph.edges(data=True):
                rt = data.get("type", "RELATED_TO")
                rel_counts[rt] = rel_counts.get(rt, 0) + 1

            return {
                "node_count": self.graph.number_of_nodes(),
                "relationship_count": self.graph.number_of_edges(),
                "node_types": [{"labels": [t], "count": c}
                               for t, c in sorted(type_counts.items(),
                                                   key=lambda x: x[1], reverse=True)],
                "relationship_types": [{"type": t, "count": c}
                                       for t, c in sorted(rel_counts.items(),
                                                          key=lambda x: x[1], reverse=True)],
            }
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {}

    def clear_graph(self) -> bool:
        if not self.connected:
            return False
        try:
            self.graph.clear()
            self._persist()
            if self.embedding_store:
                self.embedding_store.clear()
            logger.info("Graph cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            return False

    def get_graph_data(self, limit: int = 100) -> Dict[str, Any]:
        """
        Visualization data.  Seed from first *limit* nodes, include
        neighbour edges (fix from the isolated-nodes bug).
        """
        if not self.connected:
            return {"nodes": [], "edges": []}
        try:
            seed_nodes: set = set(list(self.graph.nodes())[:limit])
            edges = []
            node_set: set = set(seed_nodes)

            for u, v, data in self.graph.edges(data=True):
                if u in seed_nodes or v in seed_nodes:
                    node_set.add(u)
                    node_set.add(v)
                    edges.append({
                        "source": u, "target": v,
                        "type": data.get("type", "RELATED_TO"),
                        "properties": dict(data),
                    })

            nodes = []
            for nid in node_set:
                if not self.graph.has_node(nid):
                    continue
                d = self.graph.nodes[nid]
                nodes.append({
                    "id": nid,
                    "label": d.get("name", "Unknown"),
                    "type": d.get("type", "Unknown"),
                    "properties": dict(d),
                })

            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"Error getting graph data: {str(e)}")
            return {"nodes": [], "edges": []}

    def find_node_id(self, node_label: str) -> Optional[str]:
        """Find internal ID for a display name (fuzzy)."""
        return self._resolve_node_id(node_label)

    # ── Helpers for QueryAgent ───────────────────────────────────────────────

    def get_all_entity_names(self) -> List[str]:
        """Return display names of every node in the graph."""
        return [self.graph.nodes[n].get("name", "")
                for n in self.graph.nodes()]

    def get_all_entity_types(self) -> List[str]:
        """Return unique entity types in the graph."""
        return list({self.graph.nodes[n].get("type", "Unknown")
                     for n in self.graph.nodes()})