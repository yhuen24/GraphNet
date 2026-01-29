"""
Main module for GraphNet application.
Core functionality for building and querying the knowledge graph.
"""

import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from config import config
from document_processor import DocumentProcessor
from entity_extractor import EntityExtractor, SimpleEntityExtractor
from graph_manager import GraphManager
from query_agent import QueryAgent
from visualizer import GraphVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphNet:
    """
    Main GraphNet application class.
    Orchestrates document processing, entity extraction, and graph building.
    """
    
    def __init__(self):
        """Initialize GraphNet components"""
        self.graph_manager = GraphManager()
        self.entity_extractor = EntityExtractor()
        self.query_agent = None
        self.visualizer = GraphVisualizer()
        self.document_processor = DocumentProcessor()
        self.initialized = False
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize all components
        
        Returns:
            Dictionary with initialization status
        """
        status = {
            "graph_manager": False,
            "entity_extractor": False,
            "query_agent": False,
            "overall": False,
            "errors": []
        }
        
        try:
            # Connect to Neo4j
            if self.graph_manager.connect():
                status["graph_manager"] = True
                logger.info("✓ Graph database connected")
            else:
                status["errors"].append("Failed to connect to Neo4j")
                logger.error("✗ Graph database connection failed")
            
            # Initialize entity extractor
            if self.entity_extractor.initialize():
                status["entity_extractor"] = True
                logger.info("✓ Entity extractor initialized")
            else:
                status["errors"].append("Failed to initialize entity extractor")
                logger.warning("✗ Entity extractor initialization failed - will use fallback")
            
            # Initialize query agent
            self.query_agent = QueryAgent(self.graph_manager)
            if self.query_agent.initialize():
                status["query_agent"] = True
                logger.info("✓ Query agent initialized")
            else:
                status["errors"].append("Failed to initialize query agent")
                logger.warning("✗ Query agent initialization failed")
            
            # Check if minimum requirements are met
            status["overall"] = status["graph_manager"]
            self.initialized = status["overall"]
            
            return status
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            status["errors"].append(str(e))
            return status
    
    def process_document(self, file_bytes: bytes = None, file_path: str = None,
                        file_extension: str = None, filename: str = "unknown") -> Dict[str, Any]:
        """
        Process a document and extract entities/relationships
        
        Args:
            file_bytes: File content as bytes
            file_path: Path to file
            file_extension: File extension
            filename: Name of the file
        
        Returns:
            Processing results
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "GraphNet not initialized"
            }
        
        try:
            logger.info(f"Processing document: {filename}")
            
            # Step 1: Extract text from document
            doc_result = self.document_processor.process_file(
                file_path=file_path,
                file_bytes=file_bytes,
                file_extension=file_extension,
                filename=filename
            )
            
            if not doc_result.get("success"):
                return {
                    "success": False,
                    "error": doc_result.get("error", "Failed to process document")
                }
            
            text = doc_result.get("text", "")
            metadata = doc_result.get("metadata", {})
            
            logger.info(f"Extracted {len(text)} characters from {filename}")
            
            # Step 2: Split text into chunks
            chunks = self.document_processor.chunk_text(text)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Step 3: Extract entities and relationships
            if self.entity_extractor.initialized:
                extraction_result = self.entity_extractor.extract_from_chunks(chunks, filename)
            else:
                # Fallback to simple extraction
                logger.warning("Using fallback entity extraction")
                extraction_result = SimpleEntityExtractor.extract_basic_entities(text)
            
            entities = extraction_result.get("entities", [])
            relationships = extraction_result.get("relationships", [])
            
            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            
            # Step 4: Add to graph
            entities_added = 0
            relationships_added = 0
            
            # Add entities
            for entity in tqdm(entities, desc="Adding entities"):
                success = self.graph_manager.create_entity(
                    entity_name=entity["name"],
                    entity_type=entity["type"],
                    properties={"description": entity.get("description", "")},
                    source=filename
                )
                if success:
                    entities_added += 1
            
            # Add relationships
            for rel in tqdm(relationships, desc="Adding relationships"):
                success = self.graph_manager.create_relationship(
                    source_entity=rel["source"],
                    source_type="Entity",  # Generic type
                    target_entity=rel["target"],
                    target_type="Entity",
                    relationship_type=rel["type"].replace(" ", "_").upper(),
                    properties={"description": rel.get("description", "")}
                )
                if success:
                    relationships_added += 1
            
            logger.info(f"Added {entities_added} entities and {relationships_added} relationships to graph")
            
            return {
                "success": True,
                "filename": filename,
                "text_length": len(text),
                "chunks": len(chunks),
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "entities_added": entities_added,
                "relationships_added": relationships_added,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Query the knowledge graph using natural language
        
        Args:
            natural_language_query: User's question
        
        Returns:
            Query results
        """
        if not self.initialized or not self.query_agent:
            return {
                "success": False,
                "error": "Query agent not available"
            }
        
        return self.query_agent.process_query(natural_language_query)
    
    def get_entity_details(self, entity_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an entity
        
        Args:
            entity_name: Name of the entity
        
        Returns:
            Entity details
        """
        if not self.query_agent:
            # Fallback without query agent
            entity = self.graph_manager.get_entity(entity_name)
            relationships = self.graph_manager.get_entity_relationships(entity_name)
            
            return {
                "success": True,
                "entity": entity,
                "relationships": relationships,
                "summary": f"Entity: {entity_name}"
            }
        
        return self.query_agent.get_entity_info(entity_name)
    
    def visualize_graph(self, limit: int = 100) -> str:
        """
        Create a visualization of the graph
        
        Args:
            limit: Maximum number of nodes to visualize
        
        Returns:
            Path to HTML file
        """
        try:
            # Get graph data
            graph_data = self.graph_manager.get_graph_data(limit=limit)
            
            # Create visualization
            net = self.visualizer.create_network(graph_data)
            
            # Save to file
            filename = "graph_visualization.html"
            self.visualizer.save_html(net, filename)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph
        
        Returns:
            Graph statistics
        """
        try:
            # Get stats from database
            db_stats = self.graph_manager.get_graph_stats()
            
            # Get graph data for visualization stats
            graph_data = self.graph_manager.get_graph_data(limit=1000)
            vis_stats = self.visualizer.get_graph_statistics(graph_data)
            
            return {
                "database": db_stats,
                "visualization": vis_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def search_entities(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for entities by name
        
        Args:
            search_term: Term to search for
        
        Returns:
            List of matching entities
        """
        return self.graph_manager.search_entities(search_term)
    
    def clear_graph(self) -> bool:
        """
        Clear all data from the graph
        
        Returns:
            Success status
        """
        return self.graph_manager.clear_graph()
    
    def export_graph(self, filename: str = "graph_export.json") -> str:
        """
        Export the graph to a JSON file
        
        Args:
            filename: Output filename
        
        Returns:
            Path to exported file
        """
        graph_data = self.graph_manager.get_graph_data(limit=10000)
        return self.visualizer.export_to_json(graph_data, filename)
    
    def shutdown(self):
        """Shutdown GraphNet and close connections"""
        if self.graph_manager:
            self.graph_manager.close()
        logger.info("GraphNet shutdown complete")


# Main execution
if __name__ == "__main__":
    # Initialize GraphNet
    graphnet = GraphNet()
    
    print("=" * 60)
    print("GraphNet - AI-Powered Knowledge Graph")
    print("=" * 60)
    
    # Initialize components
    print("\nInitializing components...")
    init_status = graphnet.initialize()
    
    print("\nInitialization Status:")
    print(f"  Graph Database: {'✓' if init_status['graph_manager'] else '✗'}")
    print(f"  Entity Extractor: {'✓' if init_status['entity_extractor'] else '✗'}")
    print(f"  Query Agent: {'✓' if init_status['query_agent'] else '✗'}")
    
    if init_status['errors']:
        print("\nErrors:")
        for error in init_status['errors']:
            print(f"  - {error}")
    
    if init_status['overall']:
        print("\n✓ GraphNet initialized successfully!")
        print("\nUse the Streamlit UI (app.py) to interact with GraphNet")
    else:
        print("\n✗ GraphNet initialization failed")
        print("Please check your configuration and try again")
    
    # Cleanup
    graphnet.shutdown()
