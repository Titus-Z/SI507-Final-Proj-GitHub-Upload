"""Tests for small Streamlit app helpers."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from network_analyzer import NetworkAnalyzer
from streamlit_app import (
    add_impact_score_columns,
    build_article_impact_detail,
    build_article_option_map,
    build_node_option_map,
    build_query_tab_labels,
    build_sector_snapshot_frame,
    build_stock_leader_frame,
    build_topic_snapshot_frame,
    filter_embedding_bundle_by_tickers,
    filter_semantic_candidates,
    normalize_top_k,
    parse_ticker_text,
    semantic_features_available,
    summarize_llm_impacts,
)


def test_parse_ticker_text_normalizes_values() -> None:
    """Ticker text parsing should normalize case and drop empty values."""

    assert parse_ticker_text(" aapl, msft ,, nvda ") == ["AAPL", "MSFT", "NVDA"]


def test_normalize_top_k_disables_non_positive_values() -> None:
    """Top-k widget values should map 0 to disabled mode."""

    assert normalize_top_k(0) is None
    assert normalize_top_k(3) == 3


def test_build_query_tab_labels_excludes_optional_semantic_tab_by_default() -> None:
    """Course query tabs should only include semantic retrieval when requested."""

    labels = build_query_tab_labels(include_semantic=False)

    assert labels == [
        "Stock Search",
        "Sector Explorer",
        "Compare Stocks",
        "Path Finder",
        "Top Central Nodes",
        "News Impact",
        "New News Analysis",
    ]


def test_semantic_features_available_matches_optional_import_state() -> None:
    """The helper should reflect whether the optional semantic bundle is present."""

    assert semantic_features_available() is False


def test_build_node_option_map_creates_readable_labels() -> None:
    """Path-finder options should expose readable type-prefixed labels."""

    graph = nx.Graph()
    graph.add_node("stock:AAPL", node_type="stock", label="AAPL", ticker="AAPL")
    graph.add_node(
        "sector:Technology",
        node_type="sector",
        label="Technology",
        sector="Technology",
    )

    option_map = build_node_option_map(NetworkAnalyzer(graph))

    assert option_map["stock: AAPL"] == "stock:AAPL"
    assert option_map["sector: Technology"] == "sector:Technology"


def test_add_impact_score_columns_calculates_magnitude_and_net_score() -> None:
    """Impact rows should get explicit magnitude and signed score columns."""

    impact_frame = pd.DataFrame(
        [
            {
                "sector": "Technology",
                "impact_direction": "positive",
                "impact_strength": "high",
                "confidence": 0.5,
            },
            {
                "sector": "Energy",
                "impact_direction": "negative",
                "impact_strength": "medium",
                "confidence": 0.4,
            },
        ]
    )

    scored = add_impact_score_columns(impact_frame)

    assert scored.loc[0, "impact_magnitude"] == 1.5
    assert scored.loc[0, "net_impact_score"] == 1.5
    assert scored.loc[1, "impact_magnitude"] == 0.8
    assert scored.loc[1, "net_impact_score"] == -0.8


def test_summarize_llm_impacts_aggregates_by_entity() -> None:
    """Impact summary should aggregate row counts and net scores by entity."""

    impact_frame = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "sector": "Technology",
                "impact_direction": "positive",
                "impact_strength": "medium",
                "confidence": 0.7,
            },
            {
                "article_id": "a2",
                "sector": "Technology",
                "impact_direction": "negative",
                "impact_strength": "low",
                "confidence": 0.5,
            },
        ]
    )

    summary = summarize_llm_impacts(impact_frame, entity_column="sector")

    assert summary.loc[0, "sector"] == "Technology"
    assert summary.loc[0, "impact_rows"] == 2
    assert summary.loc[0, "article_count"] == 2
    assert round(summary.loc[0, "gross_impact_score"], 2) == 1.9
    assert round(summary.loc[0, "net_impact_score"], 2) == 0.9


def test_build_article_impact_detail_joins_sector_and_stock_text() -> None:
    """Article detail rows should include readable sector and stock impact strings."""

    summary_frame = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "event_summary": "AI expansion",
                "primary_event_type": "product",
                "scope": "sector_wide",
                "overall_market_relevance": "medium",
                "sector_impact_count": 1,
                "stock_impact_count": 1,
            }
        ]
    )
    sector_frame = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "sector": "Technology",
                "impact_direction": "positive",
                "impact_strength": "medium",
                "confidence": 0.7,
            }
        ]
    )
    stock_frame = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "ticker": "NVDA",
                "impact_direction": "positive",
                "impact_strength": "high",
                "confidence": 0.8,
            }
        ]
    )

    detail = build_article_impact_detail(summary_frame, sector_frame, stock_frame)

    assert "Technology: positive / medium / conf=0.70" in detail.loc[0, "sector_impacts"]
    assert "NVDA: positive / high / conf=0.80" in detail.loc[0, "stock_impacts"]


def test_filter_semantic_candidates_matches_title_and_summary() -> None:
    """Keyword filtering should match either the title or the summary."""

    metadata = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "title": "Apple AI push",
                "summary": "Enterprise rollout.",
                "published_at": pd.Timestamp("2026-03-20"),
                "source": "Reuters",
            },
            {
                "article_id": "a2",
                "title": "Cloud spending grows",
                "summary": "Microsoft AI revenue expands.",
                "published_at": pd.Timestamp("2026-03-19"),
                "source": "Bloomberg",
            },
        ]
    )

    filtered = filter_semantic_candidates(metadata, keyword="microsoft", limit=10)

    assert len(filtered) == 1
    assert filtered.loc[0, "article_id"] == "a2"


def test_build_article_option_map_formats_readable_labels() -> None:
    """Semantic article controls should show readable date-source-title labels."""

    metadata = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "title": "Apple AI push",
                "published_at": pd.Timestamp("2026-03-20"),
                "source": "Reuters",
            }
        ]
    )

    option_map = build_article_option_map(metadata)

    assert option_map["2026-03-20 | Reuters | Apple AI push"] == "a1"


def test_filter_embedding_bundle_by_tickers_keeps_matching_rows() -> None:
    """Ticker filtering should keep only embedding rows tied to the current scope."""

    metadata = pd.DataFrame(
        [
            {"article_id": "a1", "tickers": "AAPL, MSFT"},
            {"article_id": "a2", "tickers": "XOM"},
        ]
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])

    filtered = filter_embedding_bundle_by_tickers(
        {"metadata": metadata, "vectors": vectors},
        tickers=["AAPL"],
    )

    assert len(filtered["metadata"]) == 1
    assert filtered["metadata"].loc[0, "article_id"] == "a1"


def build_market_snapshot_analyzer() -> NetworkAnalyzer:
    """Build a small mixed graph for dashboard snapshot helper tests."""

    graph = nx.Graph()
    graph.add_node("stock:AAPL", node_type="stock", label="AAPL", ticker="AAPL", sector="Technology")
    graph.add_node("stock:MSFT", node_type="stock", label="MSFT", ticker="MSFT", sector="Technology")
    graph.add_node("stock:JPM", node_type="stock", label="JPM", ticker="JPM", sector="Financial Services")
    graph.add_node(
        "sector:Technology",
        node_type="sector",
        label="Technology",
        sector="Technology",
        stock_count=2,
    )
    graph.add_node(
        "sector:Financial Services",
        node_type="sector",
        label="Financial Services",
        sector="Financial Services",
        stock_count=1,
    )
    graph.add_node(
        "topic:technology",
        node_type="topic",
        label="technology",
        topic="technology",
        article_count=18,
    )
    graph.add_node(
        "topic:finance",
        node_type="topic",
        label="finance",
        topic="finance",
        article_count=5,
    )
    graph.add_edge("stock:AAPL", "sector:Technology", edge_type="stock_sector", weight=1.0)
    graph.add_edge("stock:MSFT", "sector:Technology", edge_type="stock_sector", weight=1.0)
    graph.add_edge("stock:JPM", "sector:Financial Services", edge_type="stock_sector", weight=1.0)
    graph.add_edge("stock:AAPL", "topic:technology", edge_type="stock_topic", weight=18.0, article_count=18)
    graph.add_edge("stock:MSFT", "topic:technology", edge_type="stock_topic", weight=8.0, article_count=8)
    graph.add_edge("stock:JPM", "topic:finance", edge_type="stock_topic", weight=5.0, article_count=5)
    graph.add_edge("stock:AAPL", "stock:MSFT", edge_type="stock_stock", weight=0.8, correlation=0.8)
    graph.add_edge(
        "sector:Technology",
        "sector:Financial Services",
        edge_type="sector_sector",
        weight=0.5,
        average_correlation=0.5,
    )
    return NetworkAnalyzer(graph)


def test_build_sector_snapshot_frame_summarizes_sector_nodes() -> None:
    """Sector snapshot helper should expose stock counts and graph degree."""

    frame = build_sector_snapshot_frame(build_market_snapshot_analyzer())

    assert frame.loc[0, "sector"] == "Technology"
    assert frame.loc[0, "stock_count"] == 2
    assert frame.loc[0, "connected_sectors"] == 1


def test_build_topic_snapshot_frame_orders_by_article_count() -> None:
    """Topic snapshot helper should rank topics by article activity."""

    frame = build_topic_snapshot_frame(build_market_snapshot_analyzer(), limit=5)

    assert frame.loc[0, "topic"] == "technology"
    assert frame.loc[0, "article_count"] == 18


def test_build_stock_leader_frame_returns_stock_centrality_rows() -> None:
    """Stock leader helper should keep only stock rows from centrality output."""

    frame = build_stock_leader_frame(build_market_snapshot_analyzer(), limit=5)

    assert set(frame.columns) >= {
        "ticker",
        "sector",
        "degree",
        "degree_centrality",
        "betweenness_centrality",
    }
    assert "AAPL" in frame["ticker"].tolist()
