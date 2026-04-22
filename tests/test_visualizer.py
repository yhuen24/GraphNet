"""
test_visualizer.py — Tests for graph/visualizer.py

Covers: visualization generation, colour mapping by entity type,
graph statistics computation, subgraph extraction, and export.
"""

import pytest
import json
import os
from unittest.mock import MagicMock
from graph.visualizer import GraphVisualizer


@pytest.fixture
def mock_graph_manager():
    """A mock graph manager that returns predictable graph data."""
    gm = MagicMock()
    gm.get_graph_data.return_value = {
        "nodes": [
            {"id": "1", "label": "HSBC Holdings", "type": "Organization",
             "properties": {"name": "HSBC Holdings", "type": "Organization"}},
            {"id": "2", "label": "Georges Elhedery", "type": "Person",
             "properties": {"name": "Georges Elhedery", "type": "Person"}},
            {"id": "3", "label": "London", "type": "Location",
             "properties": {"name": "London", "type": "Location"}},
        ],
        "edges": [
            {"source": "1", "target": "2", "type": "HAS_CEO",
             "properties": {}},
            {"source": "1", "target": "3", "type": "HEADQUARTERED_IN",
             "properties": {}},
        ],
    }
    return gm


@pytest.fixture
def visualizer(mock_graph_manager):
    return GraphVisualizer(mock_graph_manager)


@pytest.fixture
def empty_graph_manager():
    gm = MagicMock()
    gm.get_graph_data.return_value = {"nodes": [], "edges": []}
    return gm


# ═══════════════════════════════════════════════════════════════════════════
# Colour mapping
# ═══════════════════════════════════════════════════════════════════════════

class TestColourMapping:
    def test_known_types_have_colours(self, visualizer):
        assert "Person" in visualizer.colors
        assert "Organization" in visualizer.colors
        assert "Location" in visualizer.colors

    def test_unknown_type_falls_back_to_other(self, visualizer):
        assert "Other" in visualizer.colors

    def test_colours_are_valid_hex(self, visualizer):
        for colour in visualizer.colors.values():
            assert colour.startswith("#")
            assert len(colour) == 7  # #RRGGBB


# ═══════════════════════════════════════════════════════════════════════════
# Visualization generation
# ═══════════════════════════════════════════════════════════════════════════

class TestVisualization:
    def test_create_visualization_returns_filepath(self, visualizer):
        result = visualizer.create_visualization(limit=100)
        # create_visualization returns a filepath string via save_html
        assert result is not None
        assert isinstance(result, str)
        assert result.endswith(".html")
        # Clean up generated file
        if os.path.exists(result):
            os.remove(result)

    def test_visualization_with_empty_graph(self, empty_graph_manager):
        viz = GraphVisualizer(empty_graph_manager)
        try:
            result = viz.create_visualization(limit=100)
            if os.path.exists(result):
                os.remove(result)
        except Exception as e:
            pytest.fail(f"Visualization of empty graph should not crash: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Graph statistics  (method: get_graph_statistics)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphStatistics:
    def test_get_statistics_counts(self, visualizer, mock_graph_manager):
        stats = visualizer.get_graph_statistics(
            mock_graph_manager.get_graph_data()
        )
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 2

    def test_node_type_breakdown(self, visualizer, mock_graph_manager):
        stats = visualizer.get_graph_statistics(
            mock_graph_manager.get_graph_data()
        )
        assert stats["node_types"]["Organization"] == 1
        assert stats["node_types"]["Person"] == 1
        assert stats["node_types"]["Location"] == 1

    def test_relationship_type_breakdown(self, visualizer, mock_graph_manager):
        stats = visualizer.get_graph_statistics(
            mock_graph_manager.get_graph_data()
        )
        assert "HAS_CEO" in stats["relationship_types"]
        assert "HEADQUARTERED_IN" in stats["relationship_types"]

    def test_top_entities_computed(self, visualizer, mock_graph_manager):
        stats = visualizer.get_graph_statistics(
            mock_graph_manager.get_graph_data()
        )
        assert len(stats["top_entities"]) > 0
        # HSBC should be most central (has 2 edges)
        assert stats["top_entities"][0]["name"] == "HSBC Holdings"

    def test_avg_degree(self, visualizer, mock_graph_manager):
        stats = visualizer.get_graph_statistics(
            mock_graph_manager.get_graph_data()
        )
        # 2 * 2 edges / 3 nodes ≈ 1.33
        assert stats["avg_degree"] > 0

    def test_empty_graph_statistics(self, empty_graph_manager):
        viz = GraphVisualizer(empty_graph_manager)
        stats = viz.get_graph_statistics(
            empty_graph_manager.get_graph_data()
        )
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["avg_degree"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Subgraph extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestSubgraph:
    def test_create_subgraph_filters_by_name(self, visualizer, mock_graph_manager):
        graph_data = mock_graph_manager.get_graph_data()
        sub = visualizer.create_subgraph(graph_data, ["HSBC Holdings"])
        labels = [n.get("label") for n in sub["nodes"]]
        assert "HSBC Holdings" in labels

    def test_subgraph_includes_connected_edges(self, visualizer, mock_graph_manager):
        graph_data = mock_graph_manager.get_graph_data()
        sub = visualizer.create_subgraph(graph_data, ["HSBC Holdings"])
        # HSBC has 2 edges — both should be included
        assert len(sub["edges"]) == 2

    def test_subgraph_pulls_in_neighbours(self, visualizer, mock_graph_manager):
        graph_data = mock_graph_manager.get_graph_data()
        sub = visualizer.create_subgraph(graph_data, ["HSBC Holdings"])
        labels = [n.get("label") for n in sub["nodes"]]
        assert "Georges Elhedery" in labels
        assert "London" in labels

    def test_subgraph_with_no_matching_names(self, visualizer, mock_graph_manager):
        graph_data = mock_graph_manager.get_graph_data()
        sub = visualizer.create_subgraph(graph_data, ["Nonexistent Corp"])
        assert len(sub["nodes"]) == 0
        assert len(sub["edges"]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

class TestExport:
    def test_export_to_json(self, visualizer, mock_graph_manager, tmp_dir):
        graph_data = mock_graph_manager.get_graph_data()
        output_path = str(tmp_dir / "export.json")
        result = visualizer.export_to_json(graph_data, output_path)

        assert result == output_path

        with open(output_path, "r") as f:
            exported = json.load(f)
        assert len(exported["nodes"]) == 3
        assert len(exported["edges"]) == 2