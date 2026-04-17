"""Tests for interactive graph visualization helpers."""

from __future__ import annotations

import networkx as nx

from interactive_graph_app import (
    build_graph_control_frame,
    build_graph_type_frame,
    build_edge_title,
    build_node_title,
    build_top_degree_frame,
    edge_width,
    filter_graph,
    node_color,
    node_size,
)


def build_sample_graph() -> nx.Graph:
    """Create a small mixed-type graph for helper tests."""

    graph = nx.Graph()
    graph.add_node("stock:AAPL", node_type="stock", label="AAPL", ticker="AAPL")
    graph.add_node(
        "sector:Technology",
        node_type="sector",
        label="Technology",
        sector="Technology",
        stock_count=1,
    )
    graph.add_node(
        "topic:technology",
        node_type="topic",
        label="technology",
        topic="technology",
        article_count=40,
    )
    graph.add_edge(
        "stock:AAPL",
        "sector:Technology",
        edge_type="stock_sector",
        weight=1.0,
    )
    graph.add_edge(
        "stock:AAPL",
        "topic:technology",
        edge_type="stock_topic",
        weight=40,
        article_count=40,
        avg_ticker_sentiment=0.2,
    )
    return graph


def test_filter_graph_keeps_only_requested_types() -> None:
    """Filtering should remove unselected node and edge types."""

    graph = build_sample_graph()

    filtered = filter_graph(
        graph,
        included_node_types={"stock", "sector"},
        included_edge_types={"stock_sector"},
        remove_isolates=True,
    )

    assert set(filtered.nodes()) == {"stock:AAPL", "sector:Technology"}
    assert filtered.number_of_edges() == 1


def test_node_helpers_return_stable_display_values() -> None:
    """Node styling helpers should produce deterministic display values."""

    graph = build_sample_graph()
    stock_data = graph.nodes["stock:AAPL"]

    assert node_color("stock") == "#2F6BFF"
    assert node_size(stock_data, degree=2) > 0
    assert "ticker: AAPL" in build_node_title("stock:AAPL", stock_data, degree=2)


def test_edge_helpers_include_metadata() -> None:
    """Edge helpers should surface important edge fields."""

    graph = build_sample_graph()
    edge_data = graph.get_edge_data("stock:AAPL", "topic:technology")

    assert edge_width(edge_data) >= 1.0
    assert "article_count: 40" in build_edge_title(edge_data)


def test_build_graph_type_frame_summarizes_filtered_graph() -> None:
    """Type summary helper should count nodes and edges by type."""

    graph = build_sample_graph()

    node_frame = build_graph_type_frame(graph, item="node")
    edge_frame = build_graph_type_frame(graph, item="edge")

    assert set(node_frame["type"]) == {"stock", "sector", "topic"}
    assert set(edge_frame["type"]) == {"stock_sector", "stock_topic"}


def test_build_top_degree_frame_orders_nodes_by_degree() -> None:
    """Top-degree helper should expose readable labels and degree values."""

    graph = build_sample_graph()
    frame = build_top_degree_frame(graph, limit=3)

    assert frame.loc[0, "label"] == "AAPL"
    assert frame.loc[0, "degree"] == 2


def test_build_graph_control_frame_formats_values_for_display() -> None:
    """Display-control helper should expose readable yes/no and size text."""

    frame = build_graph_control_frame(
        remove_isolates=True,
        physics_enabled=False,
        show_physics_controls=True,
        height_px=760,
    )

    assert frame.to_dict("records") == [
        {"setting": "Remove isolated nodes", "value": "Yes"},
        {"setting": "Physics enabled", "value": "No"},
        {"setting": "Physics controls shown", "value": "Yes"},
        {"setting": "Graph height", "value": "760px"},
    ]
