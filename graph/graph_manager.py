"""
Graph Manager module for GraphNet.
Handles all Neo4j database operations including entity and relationship management.
"""

import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphManager:
    """Manages Neo4j graph database operations"""
    
    def __init__(self):
        """Initialize connection to Neo4j database"""
        self.driver = None
        self.connected = False
        
    def connect(self) -> bool:
        """
        Establish connection to Neo4j database
        
        Returns:
            Boolean indicating connection status
        """
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            self.connected = True
            logger.info("Successfully connected to Neo4j database")
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
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Neo4j connection closed")
    
    def create_entity(self, entity_name: str, entity_type: str, 
                     properties: Dict[str, Any] = None, source: str = None) -> bool:
        """
        Create or update an entity in the graph
        
        Args:
            entity_name: Name of the entity
            entity_type: Type/label of the entity
            properties: Additional properties
            source: Source document
        
        Returns:
            Boolean indicating success
        """
        if not self.connected:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            with self.driver.session() as session:
                query = f"""
                MERGE (e:{entity_type} {{name: $name}})
                ON CREATE SET e.created = timestamp(), e.source = $source
                ON MATCH SET e.updated = timestamp()
                SET e += $properties
                RETURN e
                """
                
                props = properties or {}
                props['name'] = entity_name
                
                session.run(query, name=entity_name, properties=props, source=source)
                return True
        except Exception as e:
            logger.error(f"Error creating entity: {str(e)}")
            return False
    
    def create_relationship(self, source_entity: str, source_type: str,
                          target_entity: str, target_type: str,
                          relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        """
        Create a relationship between two entities
        
        Args:
            source_entity: Source entity name
            source_type: Source entity type
            target_entity: Target entity name
            target_type: Target entity type
            relationship_type: Type of relationship
            properties: Additional properties
        
        Returns:
            Boolean indicating success
        """
        if not self.connected:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (source:{source_type} {{name: $source_name}})
                MATCH (target:{target_type} {{name: $target_name}})
                MERGE (source)-[r:{relationship_type}]->(target)
                ON CREATE SET r.created = timestamp()
                ON MATCH SET r.updated = timestamp()
                SET r += $properties
                RETURN r
                """
                
                props = properties or {}
                
                session.run(
                    query,
                    source_name=source_entity,
                    target_name=target_entity,
                    properties=props
                )
                return True
        except Exception as e:
            logger.error(f"Error creating relationship: {str(e)}")
            return False
    
    def query_graph(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query
        
        Args:
            query: Cypher query string
            parameters: Query parameters
        
        Returns:
            List of results
        """
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
    
    def get_entity(self, entity_name: str, entity_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entity from the graph
        
        Args:
            entity_name: Name of the entity
            entity_type: Optional type filter
        
        Returns:
            Entity data or None
        """
        if not self.connected:
            return None
        
        try:
            with self.driver.session() as session:
                if entity_type:
                    query = f"MATCH (e:{entity_type} {{name: $name}}) RETURN e"
                else:
                    query = "MATCH (e {name: $name}) RETURN e"
                
                result = session.run(query, name=entity_name)
                record = result.single()
                
                if record:
                    return dict(record['e'])
                return None
        except Exception as e:
            logger.error(f"Error getting entity: {str(e)}")
            return None
    
    def get_entity_relationships(self, entity_name: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity
        
        Args:
            entity_name: Name of the entity
            entity_type: Optional type filter
        
        Returns:
            List of relationships
        """
        if not self.connected:
            return []
        
        try:
            with self.driver.session() as session:
                if entity_type:
                    query = f"""
                    MATCH (e:{entity_type} {{name: $name}})-[r]-(other)
                    RETURN type(r) as relationship, other.name as entity, labels(other) as labels
                    """
                else:
                    query = """
                    MATCH (e {name: $name})-[r]-(other)
                    RETURN type(r) as relationship, other.name as entity, labels(other) as labels
                    """
                
                result = session.run(query, name=entity_name)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}")
            return []
    
    def search_entities(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities by name
        
        Args:
            search_term: Search term
            limit: Maximum results
        
        Returns:
            List of matching entities
        """
        if not self.connected:
            return []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (e)
                WHERE e.name CONTAINS $search_term
                RETURN e.name as name, labels(e) as types
                LIMIT $limit
                """
                
                result = session.run(query, search_term=search_term, limit=limit)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error searching entities: {str(e)}")
            return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph
        
        Returns:
            Dictionary with graph statistics
        """
        if not self.connected:
            return {}
        
        try:
            with self.driver.session() as session:
                # Count nodes
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_result.single()['count']
                
                # Count relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()['count']
                
                # Get node types
                type_result = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels, count(n) as count
                    ORDER BY count DESC
                """)
                node_types = [record.data() for record in type_result]
                
                # Get relationship types
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
                    "relationship_types": rel_types
                }
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {}
    
    def clear_graph(self) -> bool:
        """
        Clear all nodes and relationships from the graph
        
        Returns:
            Boolean indicating success
        """
        if not self.connected:
            return False
        
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Graph cleared successfully")
                return True
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            return False
    
    def get_graph_data(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get graph data for visualization
        
        Args:
            limit: Maximum number of nodes to return
        
        Returns:
            Dictionary containing nodes and edges
        """
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
                    # Process source node
                    if record['n']:
                        node_id = record['n'].element_id
                        if node_id not in nodes:
                            nodes[node_id] = {
                                'id': node_id,
                                'label': record['n'].get('name', 'Unknown'),
                                'type': list(record['n'].labels)[0] if record['n'].labels else 'Unknown',
                                'properties': dict(record['n'])
                            }
                    
                    # Process relationship and target node
                    if record['r'] and record['m']:
                        target_id = record['m'].element_id
                        if target_id not in nodes:
                            nodes[target_id] = {
                                'id': target_id,
                                'label': record['m'].get('name', 'Unknown'),
                                'type': list(record['m'].labels)[0] if record['m'].labels else 'Unknown',
                                'properties': dict(record['m'])
                            }
                        
                        edges.append({
                            'source': node_id,
                            'target': target_id,
                            'type': record['r'].type,
                            'properties': dict(record['r'])
                        })
                
                return {
                    "nodes": list(nodes.values()),
                    "edges": edges
                }
        except Exception as e:
            logger.error(f"Error getting graph data: {str(e)}")
            return {"nodes": [], "edges": []}
