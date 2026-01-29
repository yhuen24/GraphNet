"""
Embedded Graph Manager module for GraphNet.
Provides in-memory graph storage using NetworkX (no Neo4j required).
"""

import logging
import pickle
from typing import List, Dict, Any, Optional
import networkx as nx
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddedGraphManager:
    """
    Manages in-memory graph using NetworkX.
    Alternative to Neo4j - no database installation required!
    """
    
    def __init__(self, persist_file: str = "graphnet_data.pkl"):
        """Initialize the embedded graph manager"""
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.persist_file = persist_file
        self.connected = False
        
    def connect(self) -> bool:
        """
        'Connect' to the embedded graph (load from file if exists)
        
        Returns:
            Boolean indicating connection status
        """
        try:
            # Try to load existing graph
            try:
                with open(self.persist_file, 'rb') as f:
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
        """Save and close the graph"""
        if self.connected:
            self._persist()
            self.connected = False
            logger.info("Embedded graph saved and closed")
    
    def _persist(self):
        """Save graph to disk"""
        try:
            with open(self.persist_file, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Graph persisted to {self.persist_file}")
        except Exception as e:
            logger.error(f"Error persisting graph: {str(e)}")
    
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
            logger.error("Not connected to graph")
            return False
        
        try:
            # Create node ID
            node_id = f"{entity_type}:{entity_name}"
            
            # Prepare node attributes
            attrs = properties or {}
            attrs['name'] = entity_name
            attrs['type'] = entity_type
            attrs['source'] = source
            
            # Check if node exists
            if self.graph.has_node(node_id):
                # Update existing node
                self.graph.nodes[node_id].update(attrs)
                self.graph.nodes[node_id]['updated'] = datetime.now().isoformat()
            else:
                # Create new node
                attrs['created'] = datetime.now().isoformat()
                self.graph.add_node(node_id, **attrs)
            
            self._persist()  # Save after each operation
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
            logger.error("Not connected to graph")
            return False
        
        try:
            source_id = f"{source_type}:{source_entity}"
            target_id = f"{target_type}:{target_entity}"
            
            # Ensure both nodes exist
            if not self.graph.has_node(source_id):
                self.create_entity(source_entity, source_type)
            if not self.graph.has_node(target_id):
                self.create_entity(target_entity, target_type)
            
            # Add edge with properties
            attrs = properties or {}
            attrs['type'] = relationship_type
            attrs['created'] = datetime.now().isoformat()
            
            self.graph.add_edge(source_id, target_id, **attrs)
            
            self._persist()
            return True
            
        except Exception as e:
            logger.error(f"Error creating relationship: {str(e)}")
            return False
    
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
            # Try with type first
            if entity_type:
                node_id = f"{entity_type}:{entity_name}"
                if self.graph.has_node(node_id):
                    return dict(self.graph.nodes[node_id])
            
            # Search all nodes
            for node_id in self.graph.nodes():
                if self.graph.nodes[node_id].get('name') == entity_name:
                    return dict(self.graph.nodes[node_id])
            
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
            # Find the node
            node_id = None
            if entity_type:
                node_id = f"{entity_type}:{entity_name}"
            else:
                for nid in self.graph.nodes():
                    if self.graph.nodes[nid].get('name') == entity_name:
                        node_id = nid
                        break
            
            if not node_id or not self.graph.has_node(node_id):
                return []
            
            relationships = []
            
            # Outgoing edges
            for target in self.graph.successors(node_id):
                for edge_data in self.graph.get_edge_data(node_id, target).values():
                    relationships.append({
                        'relationship': edge_data.get('type', 'RELATED_TO'),
                        'entity': self.graph.nodes[target].get('name', 'Unknown'),
                        'labels': [self.graph.nodes[target].get('type', 'Unknown')],
                        'direction': 'outgoing'
                    })
            
            # Incoming edges
            for source in self.graph.predecessors(node_id):
                for edge_data in self.graph.get_edge_data(source, node_id).values():
                    relationships.append({
                        'relationship': edge_data.get('type', 'RELATED_TO'),
                        'entity': self.graph.nodes[source].get('name', 'Unknown'),
                        'labels': [self.graph.nodes[source].get('type', 'Unknown')],
                        'direction': 'incoming'
                    })
            
            return relationships
            
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
            results = []
            search_lower = search_term.lower()
            
            for node_id in self.graph.nodes():
                name = self.graph.nodes[node_id].get('name', '')
                if search_lower in name.lower():
                    results.append({
                        'name': name,
                        'types': [self.graph.nodes[node_id].get('type', 'Unknown')]
                    })
                    
                    if len(results) >= limit:
                        break
            
            return results
            
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
            # Count nodes
            node_count = self.graph.number_of_nodes()
            
            # Count edges
            edge_count = self.graph.number_of_edges()
            
            # Count by type
            type_counts = {}
            for node_id in self.graph.nodes():
                node_type = self.graph.nodes[node_id].get('type', 'Unknown')
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            
            node_types = [
                {'labels': [ntype], 'count': count}
                for ntype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            
            # Count relationship types
            rel_counts = {}
            for u, v, data in self.graph.edges(data=True):
                rel_type = data.get('type', 'RELATED_TO')
                rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
            
            rel_types = [
                {'type': rtype, 'count': count}
                for rtype, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            
            return {
                'node_count': node_count,
                'relationship_count': edge_count,
                'node_types': node_types,
                'relationship_types': rel_types
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
            self.graph.clear()
            self._persist()
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
            nodes = []
            edges = []
            
            # Get nodes (limited)
            node_list = list(self.graph.nodes())[:limit]
            
            for node_id in node_list:
                node_data = self.graph.nodes[node_id]
                nodes.append({
                    'id': node_id,
                    'label': node_data.get('name', 'Unknown'),
                    'type': node_data.get('type', 'Unknown'),
                    'properties': dict(node_data)
                })
            
            # Get edges between these nodes
            for u, v, data in self.graph.edges(data=True):
                if u in node_list and v in node_list:
                    edges.append({
                        'source': u,
                        'target': v,
                        'type': data.get('type', 'RELATED_TO'),
                        'properties': dict(data)
                    })
            
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"Error getting graph data: {str(e)}")
            return {"nodes": [], "edges": []}
    
    def query_graph(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a simple query (limited functionality compared to Cypher)
        
        This is a simplified query interface for basic operations.
        For complex queries, use the specific methods.
        
        Args:
            query: Simple query string
            parameters: Query parameters
        
        Returns:
            List of results
        """
        if not self.connected:
            return []
        
        try:
            results = []
            query_lower = query.lower()
            
            # Simple pattern matching for common queries
            if 'match (n)' in query_lower or 'show' in query_lower or 'all' in query_lower:
                # Return all nodes
                for node_id in list(self.graph.nodes())[:100]:
                    results.append({
                        'n': dict(self.graph.nodes[node_id])
                    })
            
            elif 'count' in query_lower:
                # Return count
                results.append({
                    'count': self.graph.number_of_nodes()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []
