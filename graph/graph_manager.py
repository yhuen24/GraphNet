"""
Graph Manager module for GraphNet.
Handles all Neo4j database operations including entity and relationship management.

v2 — Canonical deduplication, case-insensitive matching, embedding integration.
"""

import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from config import config
from utils.normalization import canonical_key, display_name, fuzzy_match

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphManager:
    """Manages Neo4j graph database operations"""

    def __init__(self):
        self.driver = None
        self.connected = False
        self.embedding_store = None  # set externally after init

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            self.connected = True
            logger.info("Successfully connected to Neo4j database")

            # Create indexes for canonical_name if they don't exist
            self._ensure_indexes()
            return True
        except AuthError as e:
            logger.error(f"Authentication failed: {str(e)}")
            self.connected = False
            return False
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {str(e)}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.connected = False
            return False

    def _ensure_indexes(self):
        """Create indexes for fast canonical_name lookups."""
        try:
            with self.driver.session() as session:
                # Index on canonical_name across all nodes
                session.run(
                    "CREATE INDEX idx_canonical_name IF NOT EXISTS "
                    "FOR (n:Entity) ON (n.canonical_name)"
                )
                # Full-text index for fuzzy search (Lucene-powered)
                try:
                    session.run(
                        "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                        "FOR (n:Person|Organization|Location|Concept|Product|"
                        "Date|Event|Technology|Other|Entity) "
                        "ON EACH [n.name, n.canonical_name]"
                    )
                    logger.info("Full-text index created/verified")
                except Exception:
                    # Older Neo4j versions may not support this syntax
                    logger.warning("Could not create full-text index (may need Neo4j 4.x+)")
        except Exception as e:
            logger.warning(f"Index creation skipped: {e}")

    def close(self):
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Neo4j connection closed")

    # ── Entity CRUD ──────────────────────────────────────────────────────────

    def create_entity(self, entity_name: str, entity_type: str,
                      properties: Dict[str, Any] = None,
                      source: str = None) -> bool:
        """
        Create or update an entity using canonical_name as the merge key.

        This means "Company A", "company a", and "COMPANY A" all merge
        to the same node.  The display-friendly name is stored in `name`.
        """
        if not self.connected:
            logger.error("Not connected to Neo4j")
            return False

        try:
            clean = display_name(entity_name)
            cname = canonical_key(entity_name)

            with self.driver.session() as session:
                # MERGE on canonical_name so duplicates collapse
                query = f"""
                MERGE (e:{entity_type} {{canonical_name: $canonical_name}})
                ON CREATE SET
                    e.name            = $display_name,
                    e.canonical_name  = $canonical_name,
                    e.created         = timestamp(),
                    e.source          = $source
                ON MATCH SET
                    e.updated = timestamp()
                SET e += $properties
                RETURN e
                """

                props = properties or {}
                props["name"] = clean
                props["canonical_name"] = cname

                session.run(
                    query,
                    canonical_name=cname,
                    display_name=clean,
                    properties=props,
                    source=source
                )

                # Index in embedding store
                if self.embedding_store:
                    node_id = f"{entity_type.lower()}:{cname}"
                    self.embedding_store.add_entity(
                        node_id=node_id,
                        name=clean,
                        entity_type=entity_type,
                        description=props.get("description", ""),
                    )

                return True
        except Exception as e:
            logger.error(f"Error creating entity: {str(e)}")
            return False

    def create_relationship(self, source_entity: str, source_type: str,
                            target_entity: str, target_type: str,
                            relationship_type: str,
                            properties: Dict[str, Any] = None) -> bool:
        """
        Create a relationship using case-insensitive matching on canonical_name.
        Falls back to toLower(name) CONTAINS if canonical_name isn't present
        (for backwards compatibility with pre-v2 data).
        """
        if not self.connected:
            logger.error("Not connected to Neo4j")
            return False

        try:
            src_canonical = canonical_key(source_entity)
            tgt_canonical = canonical_key(target_entity)

            with self.driver.session() as session:
                query = f"""
                MATCH (source)
                WHERE source.canonical_name = $src_cname
                   OR toLower(source.name) = $src_lower
                WITH source LIMIT 1
                MATCH (target)
                WHERE target.canonical_name = $tgt_cname
                   OR toLower(target.name) = $tgt_lower
                WITH source, target LIMIT 1
                MERGE (source)-[r:{relationship_type}]->(target)
                ON CREATE SET r.created = timestamp()
                ON MATCH SET r.updated = timestamp()
                SET r += $properties
                RETURN r
                """

                props = properties or {}

                session.run(
                    query,
                    src_cname=src_canonical,
                    src_lower=src_canonical,  # canonical_key is already lowercase
                    tgt_cname=tgt_canonical,
                    tgt_lower=tgt_canonical,
                    properties=props
                )
                return True
        except Exception as e:
            logger.error(f"Error creating relationship: {str(e)}")
            return False

    # ── Read operations (all case-insensitive) ───────────────────────────────

    def get_entity(self, entity_name: str,
                   entity_type: str = None) -> Optional[Dict[str, Any]]:
        if not self.connected:
            return None

        try:
            cname = canonical_key(entity_name)
            with self.driver.session() as session:
                if entity_type:
                    query = f"""
                    MATCH (e:{entity_type})
                    WHERE e.canonical_name = $cname
                       OR toLower(e.name) CONTAINS $cname
                    RETURN e LIMIT 1
                    """
                else:
                    query = """
                    MATCH (e)
                    WHERE e.canonical_name = $cname
                       OR toLower(e.name) CONTAINS $cname
                    RETURN e LIMIT 1
                    """

                result = session.run(query, cname=cname)
                record = result.single()
                if record:
                    return dict(record["e"])
                return None
        except Exception as e:
            logger.error(f"Error getting entity: {str(e)}")
            return None

    def get_entity_relationships(self, entity_name: str,
                                 entity_type: str = None) -> List[Dict[str, Any]]:
        if not self.connected:
            return []

        try:
            cname = canonical_key(entity_name)
            with self.driver.session() as session:
                query = """
                MATCH (e)-[r]-(other)
                WHERE e.canonical_name = $cname
                   OR toLower(e.name) CONTAINS $cname
                RETURN type(r) as relationship,
                       other.name as entity,
                       labels(other) as labels,
                       other.source as source
                LIMIT 50
                """

                result = session.run(query, cname=cname)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}")
            return []

    def search_entities(self, search_term: str,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Case-insensitive search.  Tries full-text index first (fuzzy),
        falls back to toLower() CONTAINS.
        """
        if not self.connected:
            return []

        try:
            cterm = canonical_key(search_term)
            results = []

            with self.driver.session() as session:
                # Try full-text index (supports fuzzy with ~)
                try:
                    ft_query = """
                    CALL db.index.fulltext.queryNodes(
                        'entity_fulltext', $term
                    ) YIELD node, score
                    RETURN node.name as name,
                           labels(node) as types,
                           node.source as source,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                    # Lucene fuzzy syntax: append ~ for edit distance
                    fuzzy_term = f"{search_term}~"
                    ft_result = session.run(ft_query, term=fuzzy_term, limit=limit)
                    results = [record.data() for record in ft_result]
                except Exception:
                    pass  # full-text index may not exist

                # Fallback: case-insensitive CONTAINS
                if not results:
                    fallback_query = """
                    MATCH (e)
                    WHERE toLower(e.name) CONTAINS $term
                       OR e.canonical_name CONTAINS $term
                    RETURN e.name as name,
                           labels(e) as types,
                           e.source as source
                    LIMIT $limit
                    """
                    fb_result = session.run(fallback_query, term=cterm, limit=limit)
                    results = [record.data() for record in fb_result]

            return results
        except Exception as e:
            logger.error(f"Error searching entities: {str(e)}")
            return []

    # ── Semantic search (embedding-powered) ──────────────────────────────────

    def semantic_search(self, query: str,
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Use the embedding store for deep semantic matching.
        Returns entity dicts with score.
        """
        if not self.embedding_store:
            # Fall back to string search
            return self.search_entities(query, limit=top_k)

        hits = self.embedding_store.search(query, top_k=top_k, threshold=0.20)
        results = []
        for node_id, score in hits:
            # node_id format is "type:canonical_name"
            parts = node_id.split(":", 1)
            cname = parts[1] if len(parts) > 1 else parts[0]

            # Fetch from Neo4j
            entity = self.get_entity(cname)
            if entity:
                entity["score"] = round(score, 3)
                entity["node_id"] = node_id
                results.append(entity)

        return results

    def get_neighbourhood(self, entity_names: List[str],
                          depth: int = 1) -> Dict[str, Any]:
        """
        Return the subgraph around the given entities up to *depth* hops.
        """
        if not self.connected:
            return {"nodes": [], "edges": []}

        try:
            # Build canonical names for matching
            cnames = [canonical_key(n) for n in entity_names]

            with self.driver.session() as session:
                query = f"""
                MATCH (start)
                WHERE start.canonical_name IN $cnames
                   OR toLower(start.name) IN $cnames
                CALL apoc.path.subgraphAll(start, {{maxLevel: $depth}})
                YIELD nodes, relationships
                UNWIND nodes AS n
                UNWIND relationships AS r
                RETURN DISTINCT
                    n.name AS node_name,
                    labels(n) AS node_labels,
                    n.source AS node_source,
                    n.canonical_name AS node_cname,
                    startNode(r).name AS src_name,
                    endNode(r).name AS tgt_name,
                    type(r) AS rel_type
                LIMIT 200
                """
                try:
                    result = session.run(query, cnames=cnames, depth=depth)
                    records = [r.data() for r in result]
                except Exception:
                    # APOC may not be installed — fallback to simple 1-hop
                    records = self._simple_neighbourhood(session, cnames)

            # Build nodes & edges from records
            nodes_map = {}
            edges = []
            for rec in records:
                nname = rec.get("node_name")
                if nname and nname not in nodes_map:
                    nodes_map[nname] = {
                        "id": rec.get("node_cname", canonical_key(nname)),
                        "label": nname,
                        "type": (rec.get("node_labels", ["Unknown"]) or ["Unknown"])[0],
                        "properties": {
                            "name": nname,
                            "source": rec.get("node_source", ""),
                        },
                    }
                src = rec.get("src_name")
                tgt = rec.get("tgt_name")
                rtype = rec.get("rel_type")
                if src and tgt and rtype:
                    edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rtype,
                        "properties": {},
                    })

            return {"nodes": list(nodes_map.values()), "edges": edges}
        except Exception as e:
            logger.error(f"Error getting neighbourhood: {e}")
            return {"nodes": [], "edges": []}

    def _simple_neighbourhood(self, session, cnames: List[str]) -> List[Dict]:
        """Fallback 1-hop neighbourhood without APOC."""
        query = """
        MATCH (start)-[r]-(neighbour)
        WHERE start.canonical_name IN $cnames
           OR toLower(start.name) IN $cnames
        RETURN DISTINCT
            start.name AS node_name,
            labels(start) AS node_labels,
            start.source AS node_source,
            start.canonical_name AS node_cname,
            startNode(r).name AS src_name,
            endNode(r).name AS tgt_name,
            type(r) AS rel_type
        LIMIT 200
        """
        result = session.run(query, cnames=cnames)
        records = [r.data() for r in result]

        # Also include the neighbour nodes themselves
        query2 = """
        MATCH (start)-[r]-(neighbour)
        WHERE start.canonical_name IN $cnames
           OR toLower(start.name) IN $cnames
        RETURN DISTINCT
            neighbour.name AS node_name,
            labels(neighbour) AS node_labels,
            neighbour.source AS node_source,
            neighbour.canonical_name AS node_cname,
            null AS src_name,
            null AS tgt_name,
            null AS rel_type
        LIMIT 200
        """
        result2 = session.run(query2, cnames=cnames)
        records.extend([r.data() for r in result2])
        return records

    # ── Query execution ──────────────────────────────────────────────────────

    def query_graph(self, query: str,
                    parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not self.connected:
            logger.error("Not connected to Neo4j")
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []

    # ── Stats / graph data / clear ───────────────────────────────────────────

    def get_graph_stats(self) -> Dict[str, Any]:
        if not self.connected:
            return {}

        try:
            with self.driver.session() as session:
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_result.single()["count"]

                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]

                type_result = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels, count(n) as count
                    ORDER BY count DESC
                """)
                node_types = [record.data() for record in type_result]

                rel_type_result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                    ORDER BY count DESC
                """)
                rel_types = [record.data() for record in rel_type_result]

                return {
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "node_types": node_types,
                    "relationship_types": rel_types,
                }
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {}

    def clear_graph(self) -> bool:
        if not self.connected:
            return False

        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Graph cleared successfully")
                if self.embedding_store:
                    self.embedding_store.clear()
                return True
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            return False

    def get_graph_data(self, limit: int = 100) -> Dict[str, Any]:
        if not self.connected:
            return {"nodes": [], "edges": []}

        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (n)
                WITH n LIMIT {limit}
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n, r, m
                """

                result = session.run(query)
                nodes = {}
                edges = []

                for record in result:
                    if record["n"]:
                        node_id = record["n"].element_id
                        if node_id not in nodes:
                            nodes[node_id] = {
                                "id": node_id,
                                "label": record["n"].get("name", "Unknown"),
                                "type": (list(record["n"].labels)[0]
                                         if record["n"].labels else "Unknown"),
                                "properties": dict(record["n"]),
                            }

                    if record["r"] and record["m"]:
                        target_id = record["m"].element_id
                        if target_id not in nodes:
                            nodes[target_id] = {
                                "id": target_id,
                                "label": record["m"].get("name", "Unknown"),
                                "type": (list(record["m"].labels)[0]
                                         if record["m"].labels else "Unknown"),
                                "properties": dict(record["m"]),
                            }

                        edges.append({
                            "source": node_id,
                            "target": target_id,
                            "type": record["r"].type,
                            "properties": dict(record["r"]),
                        })

                return {"nodes": list(nodes.values()), "edges": edges}
        except Exception as e:
            logger.error(f"Error getting graph data: {str(e)}")
            return {"nodes": [], "edges": []}

    # ── Helpers for QueryAgent ───────────────────────────────────────────────

    def get_all_entity_names(self) -> List[str]:
        """Return display names of every node in the graph."""
        if not self.connected:
            return []
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN n.name AS name LIMIT 5000")
                return [r["name"] for r in result if r["name"]]
        except Exception as e:
            logger.error(f"Error getting entity names: {e}")
            return []

    def get_all_entity_types(self) -> List[str]:
        """Return unique entity type labels."""
        if not self.connected:
            return []
        try:
            with self.driver.session() as session:
                result = session.run("CALL db.labels() YIELD label RETURN label")
                return [r["label"] for r in result]
        except Exception:
            return []

    def find_node_id(self, node_label: str) -> Optional[str]:
        """Find a node by display name (case-insensitive)."""
        entity = self.get_entity(node_label)
        if entity:
            return entity.get("canonical_name", canonical_key(node_label))
        return None