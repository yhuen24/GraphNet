"""
Visualizer module for GraphNet.
Handles graph visualization using PyVis and network diagrams.
"""

import logging
from typing import Dict, Any, List
from pyvis.network import Network
import networkx as nx
import json
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Visualize knowledge graphs using PyVis"""
    
    def __init__(self):
        """Initialize the graph visualizer"""
        self.colors = {
            'Person': '#FF6B6B',
            'Organization': '#4ECDC4',
            'Location': '#45B7D1',
            'Concept': '#FFA07A',
            'Product': '#98D8C8',
            'Date': '#F7DC6F',
            'Event': '#BB8FCE',
            'Technology': '#85C1E2',
            'Other': '#BDC3C7'
        }
    
    def create_network(self, graph_data: Dict[str, Any], 
                      height: str = None, width: str = None) -> Network:
        """
        Create a PyVis network from graph data
        
        Args:
            graph_data: Dictionary containing nodes and edges
            height: Height of the visualization
            width: Width of the visualization
        
        Returns:
            PyVis Network object
        """
        height = height or config.GRAPH_HEIGHT
        width = width or config.GRAPH_WIDTH
        
        # Initialize network
        net = Network(
            height=height,
            width=width,
            bgcolor='#222222',
            font_color='white',
            directed=True
        )
        
        # Configure physics
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "stabilization": {
              "enabled": true,
              "iterations": 100
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)
        
        # Add nodes
        for node in graph_data.get('nodes', []):
            node_type = node.get('type', 'Other')
            color = self.colors.get(node_type, self.colors['Other'])
            
            title = self._create_node_tooltip(node)
            
            net.add_node(
                node['id'],
                label=node.get('label', 'Unknown'),
                title=title,
                color=color,
                size=25,
                font={'size': 14}
            )
        
        # Add edges
        for edge in graph_data.get('edges', []):
            title = self._create_edge_tooltip(edge)
            
            net.add_edge(
                edge['source'],
                edge['target'],
                title=title,
                label=edge.get('type', ''),
                color={'color': '#848484'},
                arrows='to',
                font={'size': 10, 'align': 'middle'}
            )
        
        return net
    
    def _create_node_tooltip(self, node: Dict[str, Any]) -> str:
        """
        Create HTML tooltip for a node
        
        Args:
            node: Node data
        
        Returns:
            HTML string for tooltip
        """
        properties = node.get('properties', {})
        
        tooltip = f"<b>{node.get('label', 'Unknown')}</b><br>"
        tooltip += f"<i>Type: {node.get('type', 'Unknown')}</i><br><br>"
        
        # Add properties
        for key, value in properties.items():
            if key not in ['name', 'id']:
                tooltip += f"{key}: {value}<br>"
        
        return tooltip
    
    def _create_edge_tooltip(self, edge: Dict[str, Any]) -> str:
        """
        Create HTML tooltip for an edge
        
        Args:
            edge: Edge data
        
        Returns:
            HTML string for tooltip
        """
        properties = edge.get('properties', {})
        
        tooltip = f"<b>{edge.get('type', 'Unknown')}</b><br>"
        
        # Add properties
        for key, value in properties.items():
            tooltip += f"{key}: {value}<br>"
        
        return tooltip
    
    def save_html(self, net: Network, filename: str = "graph.html") -> str:
        """
        Save network visualization to HTML file
        
        Args:
            net: PyVis Network object
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        try:
            net.save_graph(filename)
            logger.info(f"Graph visualization saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            raise
    
    def create_subgraph(self, graph_data: Dict[str, Any], 
                       entity_names: List[str]) -> Dict[str, Any]:
        """
        Create a subgraph containing only specified entities and their direct connections
        
        Args:
            graph_data: Full graph data
            entity_names: List of entity names to include
        
        Returns:
            Filtered graph data
        """
        # Find nodes matching entity names
        selected_nodes = []
        selected_node_ids = set()
        
        for node in graph_data.get('nodes', []):
            if node.get('label') in entity_names:
                selected_nodes.append(node)
                selected_node_ids.add(node['id'])
        
        # Find edges connected to selected nodes
        selected_edges = []
        additional_node_ids = set()
        
        for edge in graph_data.get('edges', []):
            if edge['source'] in selected_node_ids or edge['target'] in selected_node_ids:
                selected_edges.append(edge)
                additional_node_ids.add(edge['source'])
                additional_node_ids.add(edge['target'])
        
        # Add connected nodes
        for node in graph_data.get('nodes', []):
            if node['id'] in additional_node_ids and node['id'] not in selected_node_ids:
                selected_nodes.append(node)
        
        return {
            'nodes': selected_nodes,
            'edges': selected_edges
        }
    
    def get_graph_statistics(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistics about the graph
        
        Args:
            graph_data: Graph data
        
        Returns:
            Dictionary with statistics
        """
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # Count node types
        node_type_counts = {}
        for node in nodes:
            node_type = node.get('type', 'Other')
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        # Count relationship types
        edge_type_counts = {}
        for edge in edges:
            edge_type = edge.get('type', 'Unknown')
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        # Create NetworkX graph for centrality calculations
        G = nx.DiGraph()
        
        for node in nodes:
            G.add_node(node['id'], **node)
        
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Calculate degree centrality
        if len(G.nodes()) > 0:
            centrality = nx.degree_centrality(G)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Map back to node labels
            top_entities = []
            for node_id, score in top_nodes:
                node_label = None
                for node in nodes:
                    if node['id'] == node_id:
                        node_label = node.get('label', 'Unknown')
                        break
                top_entities.append({'name': node_label, 'centrality': round(score, 3)})
        else:
            top_entities = []
        
        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'node_types': node_type_counts,
            'relationship_types': edge_type_counts,
            'top_entities': top_entities,
            'avg_degree': round(2 * len(edges) / len(nodes), 2) if len(nodes) > 0 else 0
        }
    
    def export_to_json(self, graph_data: Dict[str, Any], filename: str = "graph.json") -> str:
        """
        Export graph data to JSON file
        
        Args:
            graph_data: Graph data
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        try:
            with open(filename, 'w') as f:
                json.dump(graph_data, f, indent=2)
            logger.info(f"Graph data exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error exporting graph data: {str(e)}")
            raise
