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
    
    def __init__(self, graph_manager):
        """Initialize the graph visualizer"""
        self.graph_manager = graph_manager
        self.colors = {
            'Person': '#a89cf9',
            'Organization': '#7df3e4',
            'Location': '#fbbf24',
            'Concept': '#f472b6',
            'Product': '#4ade80',
            'Date': '#F7DC6F',
            'Event': '#BB8FCE',
            'Technology': '#60a5fa',
            'Other': '#9b9baa'
        }

    def _setup_base_network(self) -> Network:
        net = Network(height="700px", width="100%", bgcolor="#17171a",
                      font_color="#e8e8ec", directed=True)
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=250)

        net.set_options("""{
          "edges": {
            "color": {
              "color": "#5eead4",
              "highlight": "#99f6e4",
              "hover": "#99f6e4",
              "inherit": false
            },
            "font": { "color": "#e8e8ec", "size": 11, "strokeWidth": 0 },
            "width": 2,
            "arrows": {
              "to": { "enabled": true, "scaleFactor": 1.0 }
            },
            "smooth": { "type": "continuous" }
          },
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 250
            },
            "stabilization": { "iterations": 120 }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "zoomView": true
          }
        }""")
        return net

    def create_visualization(self, limit: int = 100):
        """Generates the full graph visualization."""
        graph_data = self.graph_manager.get_graph_data(limit=limit)
        net = self._setup_base_network()

        for node in graph_data.get('nodes', []):
            color = self.colors.get(node.get('type'), self.colors['Other'])
            net.add_node(node['id'], label=node.get('label'), color=color,
                         size=22, title=f"{node.get('type', '?')}: {node.get('label')}")

        for edge in graph_data.get('edges', []):
            net.add_edge(
                edge['source'], edge['target'],
                label=edge.get('type', ''),
                width=2,
                arrows="to",
            )

        return self.save_html(net, "graph_visualization.html")

    def generate_focused_visualization(self, focal_node_name: str):
        net = self._setup_base_network()

        focal_node_id = self.graph_manager.find_node_id(focal_node_name)
        if not focal_node_id or not self.graph_manager.graph.has_node(focal_node_id):
            logger.warning(f"Node '{focal_node_name}' not found. Showing full graph.")
            return self.create_visualization()

        neighbors = list(self.graph_manager.graph.successors(focal_node_id))
        predecessors = list(self.graph_manager.graph.predecessors(focal_node_id))
        nodes_to_draw = {focal_node_id} | set(neighbors) | set(predecessors)

        for node_id in nodes_to_draw:
            node_data = self.graph_manager.graph.nodes[node_id]
            is_focal = node_id == focal_node_id

            node_color = "#FFD700" if is_focal else self.colors.get(
                node_data.get("type"), "#9b9baa"
            )

            net.add_node(
                node_id,
                label=node_data.get("name", node_id),
                color=node_color,
                size=35 if is_focal else 25,
                title=f"Type: {node_data.get('type')}"
            )

        for u, v, key, data in self.graph_manager.graph.edges(keys=True, data=True):
            if u in nodes_to_draw and v in nodes_to_draw:
                net.add_edge(
                    u, v,
                    label=data.get("type", key),
                    width=2,
                    arrows="to",
                )

        output_path = "graph_visualization.html"
        net.save_graph(output_path)
        return output_path

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

        net = self._setup_base_network()

        for node in graph_data.get('nodes', []):
            node_type = node.get('type', 'Other')
            color = self.colors.get(node_type, self.colors['Other'])

            title = self._create_node_tooltip(node)

            net.add_node(
                node['id'],
                label=node.get('label', 'Unknown'),
                title=title,
                color=color,
                size=22,
                font={'size': 14}
            )

        # Add edges — white with arrows
        for edge in graph_data.get('edges', []):
            title = self._create_edge_tooltip(edge)

            net.add_edge(
                edge['source'],
                edge['target'],
                title=title,
                label=edge.get('type', ''),
                arrows='to',
                width=2,
                font={'size': 11, 'align': 'middle', 'color': '#FFFFFF'}
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
        selected_nodes = []
        selected_node_ids = set()

        for node in graph_data.get('nodes', []):
            if node.get('label') in entity_names:
                selected_nodes.append(node)
                selected_node_ids.add(node['id'])

        selected_edges = []
        additional_node_ids = set()

        for edge in graph_data.get('edges', []):
            if edge['source'] in selected_node_ids or edge['target'] in selected_node_ids:
                selected_edges.append(edge)
                additional_node_ids.add(edge['source'])
                additional_node_ids.add(edge['target'])

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

        node_type_counts = {}
        for node in nodes:
            node_type = node.get('type', 'Other')
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        edge_type_counts = {}
        for edge in edges:
            edge_type = edge.get('type', 'Unknown')
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

        G = nx.MultiDiGraph()

        for node in nodes:
            G.add_node(node['id'], **node)

        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)

        if len(G.nodes()) > 0:
            centrality = nx.degree_centrality(G)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

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