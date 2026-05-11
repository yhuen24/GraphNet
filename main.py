"""
Main module for GraphNet application.
Core functionality for building and querying the knowledge graph.

v2 — Canonical normalization, embedding store integration.
v3 — ChunkStore integration: stores original document text chunks for
     reasoning-based query answering.
"""

import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import config
from ai.document_processor import DocumentProcessor
from ai.entity_extractor import EntityExtractor, SimpleEntityExtractor
from ai.embedding_store import EmbeddingStore
from ai.chunk_store import ChunkStore
from utils.normalization import canonical_key, display_name

# Import appropriate graph manager based on configuration
if config.GRAPH_MODE == "neo4j":
    try:
        from graph.graph_manager import GraphManager
        logger.info("Using Neo4j graph manager")
    except ImportError:
        logger.warning("Neo4j not available, falling back to embedded mode")
        from graph.embedded_graph_manager import EmbeddedGraphManager as GraphManager
else:
    from graph.embedded_graph_manager import EmbeddedGraphManager as GraphManager
    logger.info("Using embedded graph manager (no database needed)")

from ai.query_agent import QueryAgent
from graph.visualizer import GraphVisualizer


class GraphNet:
    """
    Main GraphNet application class.
    Orchestrates document processing, entity extraction, and graph building.
    """

    def __init__(self):
        self.graph_manager = GraphManager()
        self.entity_extractor = EntityExtractor()
        self.embedding_store = EmbeddingStore()
        self.chunk_store = ChunkStore()         # NEW: document chunk storage
        self.query_agent = None
        self.visualizer = GraphVisualizer(self.graph_manager)
        self.document_processor = DocumentProcessor()
        self.initialized = False

    def initialize(self) -> Dict[str, Any]:
        status = {
            "graph_manager": False,
            "entity_extractor": False,
            "query_agent": False,
            "embedding_store": False,
            "chunk_store": False,               # NEW
            "overall": False,
            "errors": [],
        }

        try:
            # 1. Connect graph
            if self.graph_manager.connect():
                status["graph_manager"] = True
                logger.info("✓ Graph database connected")
            else:
                status["errors"].append("Failed to connect to graph")
                logger.error("✗ Graph database connection failed")

            # 2. Initialize embedding store
            if self.embedding_store.initialize(google_api_key=config.GOOGLE_API_KEY):
                status["embedding_store"] = True
                logger.info("✓ Embedding store initialized")
                # Wire the embedding store into the graph manager
                if hasattr(self.graph_manager, "embedding_store"):
                    self.graph_manager.embedding_store = self.embedding_store
                    # Re-index existing entities that may not have embeddings yet
                    self._reindex_existing_entities()
            else:
                status["errors"].append("Embedding store init failed (semantic search disabled)")
                logger.warning("✗ Embedding store failed — fuzzy search still works")

            # 3. Initialize chunk store (NEW)
            if self.chunk_store.initialize(google_api_key=config.GOOGLE_API_KEY):
                status["chunk_store"] = True
                logger.info(f"✓ Chunk store initialized ({self.chunk_store.count()} chunks)")
            else:
                status["errors"].append("Chunk store init failed (document context disabled)")
                logger.warning("✗ Chunk store failed — queries will work without document context")

            # 4. Entity extractor
            if self.entity_extractor.initialize():
                status["entity_extractor"] = True
                logger.info("✓ Entity extractor initialized")
            else:
                status["errors"].append("Entity extractor init failed — will use fallback")
                logger.warning("✗ Entity extractor initialization failed")

            # 5. Query agent — now receives chunk_store for reasoning
            self.query_agent = QueryAgent(
                self.graph_manager,
                chunk_store=self.chunk_store if status["chunk_store"] else None,
            )
            if self.query_agent.initialize():
                status["query_agent"] = True
                logger.info("✓ Query agent initialized")
            else:
                status["errors"].append("Query agent init failed")
                logger.warning("✗ Query agent initialization failed")

            status["overall"] = status["graph_manager"]
            self.initialized = status["overall"]
            return status

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            status["errors"].append(str(e))
            return status

    def _reindex_existing_entities(self):
        """Index any graph nodes that don't yet have embeddings."""
        if not self.embedding_store:
            return

        existing_ids = {e["node_id"] for e in self.embedding_store.entries}
        batch = []

        if hasattr(self.graph_manager, "graph"):
            # Embedded mode — iterate NetworkX graph directly
            for nid in self.graph_manager.graph.nodes():
                if nid in existing_ids:
                    continue
                nd = self.graph_manager.graph.nodes[nid]
                batch.append({
                    "node_id": nid,
                    "name": nd.get("name", ""),
                    "entity_type": nd.get("type", ""),
                    "description": nd.get("description", ""),
                })
        elif hasattr(self.graph_manager, "driver") and self.graph_manager.connected:
            # Neo4j mode — query the database
            try:
                from utils.normalization import canonical_key
                with self.graph_manager.driver.session() as session:
                    result = session.run("""
                        MATCH (n)
                        RETURN n.name AS name,
                               labels(n) AS labels,
                               n.description AS description,
                               n.canonical_name AS cname
                        LIMIT 5000
                    """)
                    for record in result:
                        name = record["name"] or ""
                        etype = (record["labels"][0]
                                 if record["labels"] else "Unknown")
                        cname = record["cname"] or canonical_key(name)
                        node_id = f"{etype.lower()}:{cname}"
                        if node_id in existing_ids:
                            continue
                        batch.append({
                            "node_id": node_id,
                            "name": name,
                            "entity_type": etype,
                            "description": record["description"] or "",
                        })
            except Exception as e:
                logger.warning(f"Could not re-index from Neo4j: {e}")

        if batch:
            logger.info(f"Re-indexing {len(batch)} entities into embedding store…")
            self.embedding_store.add_entities_batch(batch)
            self.embedding_store.save()

    # ── Document processing ──────────────────────────────────────────────────

    def process_document(self, file_bytes: bytes = None, file_path: str = None,
                         file_extension: str = None,
                         filename: str = "unknown") -> Dict[str, Any]:
        if not self.initialized:
            return {"success": False, "error": "GraphNet not initialized"}

        try:
            logger.info(f"Processing document: {filename}")

            # Step 1: Extract text
            doc_result = self.document_processor.process_file(
                file_path=file_path,
                file_bytes=file_bytes,
                file_extension=file_extension,
                filename=filename,
            )
            if not doc_result.get("success"):
                return {"success": False,
                        "error": doc_result.get("error", "Failed to process document")}

            text = doc_result.get("text", "")
            metadata = doc_result.get("metadata", {})
            logger.info(f"Extracted {len(text)} characters from {filename}")

            # Step 2: Chunk
            chunks = self.document_processor.chunk_text(text)
            logger.info(f"Split into {len(chunks)} chunks")

            # ══════════════════════════════════════════════════════════════
            # Store chunks for later retrieval by QueryAgent
            # ══════════════════════════════════════════════════════════════
            chunks_stored = 0
            if self.chunk_store:
                try:
                    chunks_stored = self.chunk_store.store_chunks(chunks, filename)
                    logger.info(f"Stored {chunks_stored} chunks for document context")
                except Exception as e:
                    logger.warning(f"Chunk storage failed (non-fatal): {e}")

            # Step 3: Extract entities & relationships
            if self.entity_extractor.initialized:
                extraction_result = self.entity_extractor.extract_from_chunks(
                    chunks, filename
                )
            else:
                logger.warning("Using fallback entity extraction")
                extraction_result = SimpleEntityExtractor.extract_basic_entities(text)

            entities = extraction_result.get("entities", [])
            relationships = extraction_result.get("relationships", [])
            logger.info(f"Extracted {len(entities)} entities and "
                        f"{len(relationships)} relationships")

            # Step 4: Normalize and add to graph
            entities_added = 0
            relationships_added = 0

            for entity in tqdm(entities, desc="Adding entities"):
                clean = display_name(entity["name"])

                # Merge ALL extracted properties (value, unit, period, etc.)
                # with the description so they're stored on the graph node.
                props = dict(entity.get("properties", {}))
                props["description"] = entity.get("description", "")

                success = self.graph_manager.create_entity(
                    entity_name=clean,
                    entity_type=entity["type"],
                    properties=props,
                    source=filename,
                )
                if success:
                    entities_added += 1

            for rel in tqdm(relationships, desc="Adding relationships"):
                source_clean = display_name(rel["source"])
                target_clean = display_name(rel["target"])

                rel_props = dict(rel.get("properties", {}))
                rel_props["description"] = rel.get("description", "")

                success = self.graph_manager.create_relationship(
                    source_entity=source_clean,
                    source_type="Entity",
                    target_entity=target_clean,
                    target_type="Entity",
                    relationship_type=rel["type"].replace(" ", "_").upper(),
                    properties=rel_props,
                )
                if success:
                    relationships_added += 1

            # Persist embeddings after batch
            if self.embedding_store:
                self.embedding_store.save()

            logger.info(f"Added {entities_added} entities and "
                        f"{relationships_added} relationships to graph")

            return {
                "success": True,
                "filename": filename,
                "text_length": len(text),
                "chunks": len(chunks),
                "chunks_stored": chunks_stored,     # NEW
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "entities_added": entities_added,
                "relationships_added": relationships_added,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"success": False, "error": str(e)}

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, natural_language_query: str) -> Dict[str, Any]:
        if not self.initialized or not self.query_agent:
            return {"success": False, "error": "Query agent not available"}
        return self.query_agent.process_query(natural_language_query)

    def get_entity_details(self, entity_name: str) -> Dict[str, Any]:
        if not self.query_agent:
            entity = self.graph_manager.get_entity(entity_name)
            relationships = self.graph_manager.get_entity_relationships(entity_name)
            return {
                "success": True,
                "entity": entity,
                "relationships": relationships,
                "summary": f"Entity: {entity_name}",
            }
        return self.query_agent.get_entity_info(entity_name)

    # ── Visualization ────────────────────────────────────────────────────────

    def visualize_graph(self, limit: int = 100) -> str:
        try:
            graph_data = self.graph_manager.get_graph_data(limit=limit)
            net = self.visualizer.create_network(graph_data)
            filename = "graph_visualization.html"
            self.visualizer.save_html(net, filename)
            return filename
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise

    def get_graph_statistics(self) -> Dict[str, Any]:
        try:
            db_stats = self.graph_manager.get_graph_stats()
            graph_data = self.graph_manager.get_graph_data(limit=1000)
            vis_stats = self.visualizer.get_graph_statistics(graph_data)
            return {"database": db_stats, "visualization": vis_stats}
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}

    def search_entities(self, search_term: str) -> List[Dict[str, Any]]:
        return self.graph_manager.search_entities(search_term)

    def visualize_focused(self, node_name: str):
        return self.visualizer.generate_focused_visualization(node_name)

    def clear_graph(self) -> bool:
        # Also clear chunks when graph is cleared
        if self.chunk_store:
            self.chunk_store.clear()
        return self.graph_manager.clear_graph()

    def export_graph(self, filename: str = "graph_export.json") -> str:
        graph_data = self.graph_manager.get_graph_data(limit=10000)
        return self.visualizer.export_to_json(graph_data, filename)

    def shutdown(self):
        if self.graph_manager:
            self.graph_manager.close()
        if self.embedding_store:
            self.embedding_store.save()
        if self.chunk_store:
            self.chunk_store.save()
        logger.info("GraphNet shutdown complete")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graphnet = GraphNet()
    print("=" * 60)
    print("GraphNet - AI-Powered Knowledge Graph")
    print("=" * 60)

    print("\nInitializing components...")
    init_status = graphnet.initialize()

    print("\nInitialization Status:")
    print(f"  Graph Database:    {'✓' if init_status['graph_manager'] else '✗'}")
    print(f"  Embedding Store:   {'✓' if init_status['embedding_store'] else '✗'}")
    print(f"  Chunk Store:       {'✓' if init_status.get('chunk_store') else '✗'}")
    print(f"  Entity Extractor:  {'✓' if init_status['entity_extractor'] else '✗'}")
    print(f"  Query Agent:       {'✓' if init_status['query_agent'] else '✗'}")

    if init_status["errors"]:
        print("\nWarnings:")
        for error in init_status["errors"]:
            print(f"  - {error}")

    if init_status["overall"]:
        print("\n✓ GraphNet initialized successfully!")
        print("\nUse the Streamlit UI (app.py) to interact with GraphNet")
    else:
        print("\n✗ GraphNet initialization failed")

    graphnet.shutdown()