"""Minimal Streamlit interface for local market graph exploration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from cli import build_analyzer_from_local_data
from llm_news_impact_analyzer import (
    LLMNewsImpactAnalyzer,
    LLMNewsImpactAnalyzerError,
    NewNewsImpactAssessment,
    build_assessment_frames,
    build_graph_grounding_frames,
    extract_graph_universe,
)
from local_data_store import LocalDataStore, LocalDataStoreError
from news_graph_augmenter import NewsGraphAugmenter
from network_analyzer import NetworkAnalyzer, NetworkAnalyzerError
try:
    from semantic_retriever import SemanticNewsRetriever, SemanticRetrieverError
except ImportError:  # pragma: no cover - optional feature in course-only bundles.
    SemanticNewsRetriever = None

    class SemanticRetrieverError(Exception):
        """Fallback error type when semantic retrieval code is unavailable."""

try:
    import streamlit as st
except ImportError:  # pragma: no cover - handled at runtime when Streamlit is missing.
    st = None


VALID_TOPIC_WEIGHTS = [
    "article_count",
    "avg_topic_relevance",
    "avg_ticker_relevance",
    "avg_ticker_sentiment",
    "avg_overall_sentiment",
]
IMPACT_STRENGTH_WEIGHTS = {"low": 1.0, "medium": 2.0, "high": 3.0}
IMPACT_DIRECTION_SIGNS = {
    "positive": 1.0,
    "negative": -1.0,
    "mixed": 0.0,
    "uncertain": 0.0,
}
# This default is intentionally larger than the original 5-stock demo so the
# first web view shows multiple sectors and a richer topic layer.
DEFAULT_WEB_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "ORCL",
    "AMZN",
    "TSLA",
    "JPM",
    "BAC",
    "UNH",
    "JNJ",
    "XOM",
    "WMT",
    "DIS",
]

APP_ACCENT_COLORS = [
    "#0F766E",
    "#2563EB",
    "#D97706",
    "#9333EA",
    "#BE123C",
    "#4F46E5",
]


def semantic_features_available() -> bool:
    """Return whether optional semantic retrieval helpers are bundled locally."""

    return SemanticNewsRetriever is not None


def parse_ticker_text(raw_value: str) -> list[str] | None:
    """Normalize a comma-separated ticker string into a list."""

    items = [item.strip().upper() for item in raw_value.split(",")]
    clean_items = [item for item in items if item]
    return clean_items or None


def normalize_top_k(raw_value: int) -> int | None:
    """Convert a numeric top-k widget value into builder input."""

    if raw_value <= 0:
        return None
    return raw_value


def inject_query_app_styles() -> None:
    """Apply a lightweight dashboard theme to the query-focused Streamlit app."""

    if st is None:
        return

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 26%),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.08), transparent 24%),
                linear-gradient(180deg, #F6F8FC 0%, #EEF3F8 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1340px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        section[data-testid="stSidebar"] * {
            color: #E5EEF8;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: rgba(255, 255, 255, 0.65);
            padding: 0.35rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.20);
        }
        button[data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
            background: transparent;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%);
            color: white;
        }
        .market-hero {
            padding: 1.35rem 1.45rem;
            border-radius: 24px;
            color: white;
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.96) 0%, rgba(37, 99, 235, 0.96) 100%);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.16);
            margin-bottom: 1rem;
        }
        .market-hero h2 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
        }
        .market-hero p {
            margin: 0;
            max-width: 900px;
            font-size: 1rem;
            line-height: 1.55;
            color: rgba(255, 255, 255, 0.92);
        }
        .section-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }
        .section-card h3 {
            margin-top: 0;
            margin-bottom: 0.25rem;
            font-size: 1.08rem;
        }
        .section-card p {
            margin: 0;
            color: #475569;
            line-height: 1.55;
        }
        .snapshot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.9rem;
            margin-bottom: 1rem;
        }
        .snapshot-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .snapshot-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 0.35rem;
        }
        .snapshot-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.3rem;
        }
        .snapshot-meta {
            font-size: 0.9rem;
            color: #475569;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }
        .badge-pill {
            border-radius: 999px;
            padding: 0.32rem 0.7rem;
            font-size: 0.82rem;
            background: rgba(15, 118, 110, 0.10);
            color: #0F766E;
            border: 1px solid rgba(15, 118, 110, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_query_tab_labels(include_semantic: bool) -> list[str]:
    """Build the tab list for the course query app."""

    labels = [
        "Stock Search",
        "Sector Explorer",
        "Compare Stocks",
        "Path Finder",
        "Top Central Nodes",
        "News Impact",
        "New News Analysis",
    ]
    if include_semantic:
        labels.append("Semantic News")
    return labels


def counter_to_frame(counter_data) -> pd.DataFrame:
    """Convert one counter-like object into a small display DataFrame."""

    rows = [
        {"type": item_type, "count": count}
        for item_type, count in sorted(counter_data.items())
    ]
    return pd.DataFrame(rows)


def rows_to_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Convert a row list into a DataFrame for Streamlit tables."""

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_sector_snapshot_frame(analyzer: NetworkAnalyzer) -> pd.DataFrame:
    """Build one sector-level overview table for dashboard-style summaries."""

    rows: list[dict[str, Any]] = []
    graph = analyzer.graph
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("node_type") != "sector":
            continue

        neighbor_ids = list(graph.neighbors(node_id))
        connected_stock_count = sum(
            1
            for neighbor_id in neighbor_ids
            if graph.nodes[neighbor_id].get("node_type") == "stock"
        )
        connected_sector_count = sum(
            1
            for neighbor_id in neighbor_ids
            if graph.nodes[neighbor_id].get("node_type") == "sector"
        )
        rows.append(
            {
                "sector": node_data.get("sector"),
                "stock_count": int(node_data.get("stock_count") or connected_stock_count),
                "graph_degree": graph.degree(node_id),
                "connected_sectors": connected_sector_count,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["sector", "stock_count", "graph_degree", "connected_sectors"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["stock_count", "graph_degree", "sector"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def build_topic_snapshot_frame(
    analyzer: NetworkAnalyzer,
    limit: int = 8,
) -> pd.DataFrame:
    """Build one topic activity summary for the dashboard overview."""

    rows: list[dict[str, Any]] = []
    graph = analyzer.graph
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("node_type") != "topic":
            continue

        connected_stocks = sum(
            1
            for neighbor_id in graph.neighbors(node_id)
            if graph.nodes[neighbor_id].get("node_type") == "stock"
        )
        rows.append(
            {
                "topic": node_data.get("topic"),
                "article_count": int(node_data.get("article_count") or 0),
                "connected_stocks": connected_stocks,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["topic", "article_count", "connected_stocks"])

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["article_count", "connected_stocks", "topic"],
            ascending=[False, False, True],
        )
        .head(limit)
        .reset_index(drop=True)
    )


def build_stock_leader_frame(
    analyzer: NetworkAnalyzer,
    limit: int = 8,
) -> pd.DataFrame:
    """Build one stock leadership table for the overview dashboard."""

    centrality = analyzer.compute_centrality_metrics()
    if centrality.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "sector",
                "degree",
                "degree_centrality",
                "betweenness_centrality",
            ]
        )

    graph = analyzer.graph
    rows: list[dict[str, Any]] = []
    for _, row in centrality.iterrows():
        node_id = row["node_id"]
        node_data = graph.nodes[node_id]
        if node_data.get("node_type") != "stock":
            continue
        rows.append(
            {
                "ticker": node_data.get("ticker"),
                "sector": node_data.get("sector"),
                "degree": int(graph.degree(node_id)),
                "degree_centrality": float(row["degree_centrality"]),
                "betweenness_centrality": float(row["betweenness_centrality"]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "sector",
                "degree",
                "degree_centrality",
                "betweenness_centrality",
            ]
        )

    return pd.DataFrame(rows).head(limit).reset_index(drop=True)


def render_snapshot_cards(frame: pd.DataFrame) -> None:
    """Render a compact sector card grid inspired by market overview dashboards."""

    if frame.empty:
        st.info("No sector snapshot is available for the current graph.")
        return

    rows = list(frame.head(6).iterrows())
    for start_index in range(0, len(rows), 3):
        visible_rows = rows[start_index : start_index + 3]
        columns = st.columns(len(visible_rows))
        for column, (index, row) in zip(columns, visible_rows):
            accent = APP_ACCENT_COLORS[index % len(APP_ACCENT_COLORS)]
            with column:
                st.markdown(
                    f"""
                    <div class="snapshot-card" style="border-top: 4px solid {accent};">
                        <div class="snapshot-label">Sector</div>
                        <div class="snapshot-value">{row['sector']}</div>
                        <div class="snapshot-meta">
                            stocks: {int(row['stock_count'])}<br>
                            graph degree: {int(row['graph_degree'])}<br>
                            linked sectors: {int(row['connected_sectors'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_market_overview(
    analyzer: NetworkAnalyzer,
    summary: dict[str, Any],
    context: dict[str, Any],
) -> None:
    """Render a market-overview section inspired by financial dashboard layouts."""

    sector_snapshot = build_sector_snapshot_frame(analyzer)
    topic_snapshot = build_topic_snapshot_frame(analyzer)
    stock_leaders = build_stock_leader_frame(analyzer)

    st.markdown(
        """
        <div class="market-hero">
            <h2>Graph-First Market Intelligence Workspace</h2>
            <p>
                This page is designed as a compact market dashboard: a broad overview
                first, then focused graph queries. The layout is intentionally closer
                to modern market research tools, where you scan the market structure
                before drilling into one stock, sector, path, or news event.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Tracked Stocks", summary["node_counts"].get("stock", 0))
    metric_columns[1].metric("Sectors", summary["node_counts"].get("sector", 0))
    metric_columns[2].metric("Topics", summary["node_counts"].get("topic", 0))
    metric_columns[3].metric("Articles", context["article_count"])

    st.markdown(
        """
        <div class="section-card">
            <h3>Market Snapshot</h3>
            <p>
                Sector cards mimic the fast scan pattern used in financial dashboards.
                The topic and leadership panels then show where the current graph is
                densest.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_snapshot_cards(sector_snapshot)

    left_column, right_column = st.columns([1.2, 1.0])
    with left_column:
        st.caption("Top Topics by Article Count")
        if topic_snapshot.empty:
            st.write("No topic data available.")
        else:
            st.bar_chart(
                topic_snapshot.set_index("topic")[["article_count", "connected_stocks"]]
            )
            st.dataframe(topic_snapshot, use_container_width=True, hide_index=True)

    with right_column:
        st.caption("Top Stock Leaders")
        if stock_leaders.empty:
            st.write("No stock leader data available.")
        else:
            leader_chart = stock_leaders.set_index("ticker")[
                ["degree_centrality", "betweenness_centrality"]
            ]
            st.bar_chart(leader_chart)
            st.dataframe(stock_leaders, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="section-card">
            <h3>How To Read This Page</h3>
            <p>
                Use <strong>News Impact</strong> for offline LLM labels over historical
                articles. Use <strong>New News Analysis</strong> when you want to enter
                a fresh headline and assess likely topic, sector, and stock exposure
                inside the current graph universe.
            </p>
            <div class="badge-row">
                <span class="badge-pill">overview before query</span>
                <span class="badge-pill">graph-grounded analysis</span>
                <span class="badge-pill">LLM used for impact assessment, not prediction</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_course_guide() -> None:
    """Render a short course-focused usage guide in the sidebar."""

    st.markdown(
        """
        <div class="section-card" style="background: rgba(15, 23, 42, 0.18); border-color: rgba(148, 163, 184, 0.12);">
            <h3 style="color: #F8FAFC;">Course Web Flow</h3>
            <p style="color: #DBEAFE;">
                1. Scan the overview.<br>
                2. Use <strong>Stock Search</strong> or <strong>Compare Stocks</strong>.<br>
                3. Use <strong>News Impact</strong> for historical LLM labels.<br>
                4. Use <strong>New News Analysis</strong> for one new article.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_impact_score_columns(impact_frame: pd.DataFrame) -> pd.DataFrame:
    """Add explicit magnitude and signed score columns to one impact table."""

    if impact_frame.empty:
        return impact_frame.copy()

    scored = impact_frame.copy()
    scored["strength_weight"] = (
        scored["impact_strength"].map(IMPACT_STRENGTH_WEIGHTS).fillna(0.0)
    )
    scored["direction_sign"] = (
        scored["impact_direction"].map(IMPACT_DIRECTION_SIGNS).fillna(0.0)
    )
    scored["confidence"] = pd.to_numeric(scored["confidence"], errors="coerce").fillna(0.0)
    scored["impact_magnitude"] = scored["strength_weight"] * scored["confidence"]
    scored["net_impact_score"] = scored["direction_sign"] * scored["impact_magnitude"]
    return scored


def summarize_llm_impacts(
    impact_frame: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """Aggregate article-level impact rows into one entity summary table."""

    if impact_frame.empty:
        return pd.DataFrame(
            columns=[
                entity_column,
                "impact_rows",
                "article_count",
                "net_impact_score",
                "gross_impact_score",
                "avg_confidence",
                "positive_count",
                "negative_count",
                "mixed_count",
                "uncertain_count",
            ]
        )

    scored = add_impact_score_columns(impact_frame)
    summary = (
        scored.groupby(entity_column, as_index=False)
        .agg(
            impact_rows=("article_id", "size"),
            article_count=("article_id", "nunique"),
            net_impact_score=("net_impact_score", "sum"),
            gross_impact_score=("impact_magnitude", "sum"),
            avg_confidence=("confidence", "mean"),
            positive_count=("impact_direction", lambda s: int((s == "positive").sum())),
            negative_count=("impact_direction", lambda s: int((s == "negative").sum())),
            mixed_count=("impact_direction", lambda s: int((s == "mixed").sum())),
            uncertain_count=("impact_direction", lambda s: int((s == "uncertain").sum())),
        )
        .sort_values(
            ["gross_impact_score", "net_impact_score", entity_column],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    return summary


def format_impact_list(
    impact_frame: pd.DataFrame,
    entity_column: str,
) -> dict[str, str]:
    """Convert article-level impact rows into compact per-article text strings."""

    if impact_frame.empty:
        return {}

    grouped: dict[str, str] = {}
    for article_id, article_rows in impact_frame.groupby("article_id", sort=False):
        parts = []
        for _, row in article_rows.iterrows():
            parts.append(
                f"{row[entity_column]}: {row['impact_direction']} / "
                f"{row['impact_strength']} / conf={float(row['confidence']):.2f}"
            )
        grouped[str(article_id)] = " | ".join(parts)
    return grouped


def build_article_impact_detail(
    summary_frame: pd.DataFrame,
    sector_frame: pd.DataFrame,
    stock_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Combine article summaries with sector and stock impact text for display."""

    if summary_frame.empty:
        return pd.DataFrame()

    sector_map = format_impact_list(sector_frame, entity_column="sector")
    stock_map = format_impact_list(stock_frame, entity_column="ticker")
    detail = summary_frame.copy()
    detail["sector_impacts"] = detail["article_id"].map(sector_map).fillna("")
    detail["stock_impacts"] = detail["article_id"].map(stock_map).fillna("")
    detail["sector_impact_count"] = pd.to_numeric(
        detail["sector_impact_count"], errors="coerce"
    ).fillna(0).astype(int)
    detail["stock_impact_count"] = pd.to_numeric(
        detail["stock_impact_count"], errors="coerce"
    ).fillna(0).astype(int)
    return detail[
        [
            "article_id",
            "event_summary",
            "primary_event_type",
            "scope",
            "overall_market_relevance",
            "sector_impact_count",
            "stock_impact_count",
            "sector_impacts",
            "stock_impacts",
        ]
    ]


def filter_llm_impact_tables(
    llm_tables: dict[str, pd.DataFrame],
    tickers: list[str] | None,
    sector_options: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Filter LLM impact tables to the current graph scope where possible."""

    if tickers is None and not sector_options:
        return llm_tables

    summary_frame = llm_tables["article_llm_summary"]
    sector_frame = llm_tables["article_sector_impacts"]
    stock_frame = llm_tables["article_stock_impacts"]

    filtered_stock = stock_frame.copy()
    filtered_sector = sector_frame.copy()

    if tickers is not None and not filtered_stock.empty:
        filtered_stock = filtered_stock[filtered_stock["ticker"].isin(tickers)].reset_index(
            drop=True
        )

    if sector_options and not filtered_sector.empty:
        filtered_sector = filtered_sector[
            filtered_sector["sector"].isin(sector_options)
        ].reset_index(drop=True)

    selected_article_ids = set(filtered_stock.get("article_id", []).tolist()) | set(
        filtered_sector.get("article_id", []).tolist()
    )
    if not selected_article_ids:
        filtered_summary = summary_frame.iloc[0:0].copy()
    else:
        filtered_summary = summary_frame[
            summary_frame["article_id"].isin(selected_article_ids)
        ].reset_index(drop=True)

    return {
        "article_llm_summary": filtered_summary,
        "article_sector_impacts": filtered_sector,
        "article_stock_impacts": filtered_stock,
    }


def build_node_option_map(analyzer: NetworkAnalyzer) -> dict[str, str]:
    """Build readable node labels for the path-finder controls."""

    option_map: dict[str, str] = {}
    # Keep the display label readable while still preserving the exact node id
    # that the analyzer expects under the hood.
    for node_id, node_data in sorted(analyzer.graph.nodes(data=True)):
        node_type = node_data.get("node_type", "unknown")
        label = node_data.get("label") or node_id
        option_map[f"{node_type}: {label}"] = node_id
    return option_map


def filter_embedding_bundle_by_tickers(
    embedding_bundle: dict[str, Any],
    tickers: list[str] | None,
) -> dict[str, Any]:
    """Filter an embedding bundle to articles connected to the selected tickers."""

    if tickers is None:
        return embedding_bundle

    metadata = embedding_bundle["metadata"]
    vectors = embedding_bundle["vectors"]
    if metadata.empty or "tickers" not in metadata.columns:
        return embedding_bundle

    selected_tickers = {ticker.strip().upper() for ticker in tickers if ticker.strip()}
    if not selected_tickers:
        return embedding_bundle

    mask = metadata["tickers"].fillna("").apply(
        lambda value: bool(
            selected_tickers
            & {
                item.strip().upper()
                for item in str(value).split(",")
                if item.strip()
            }
        )
    )
    filtered_metadata = metadata[mask].reset_index(drop=True)
    filtered_vectors = vectors[mask.to_numpy()]
    return {
        "metadata": filtered_metadata,
        "vectors": filtered_vectors,
    }


def filter_semantic_candidates(
    metadata: pd.DataFrame,
    keyword: str,
    limit: int = 200,
) -> pd.DataFrame:
    """Filter source-article candidates for the semantic retrieval controls."""

    candidates = metadata.copy()
    clean_keyword = keyword.strip()
    if clean_keyword:
        title_mask = candidates["title"].fillna("").str.contains(
            clean_keyword,
            case=False,
            regex=False,
        )
        summary_mask = candidates["summary"].fillna("").str.contains(
            clean_keyword,
            case=False,
            regex=False,
        )
        candidates = candidates[title_mask | summary_mask]

    candidates = candidates.sort_values(
        "published_at",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    return candidates.head(limit).reset_index(drop=True)


def build_article_option_map(metadata: pd.DataFrame) -> dict[str, str]:
    """Build readable article labels for semantic search controls."""

    option_map: dict[str, str] = {}
    for _, row in metadata.iterrows():
        published_at = row.get("published_at")
        if pd.notna(published_at):
            date_prefix = pd.Timestamp(published_at).strftime("%Y-%m-%d")
        else:
            date_prefix = "unknown-date"

        source = str(row.get("source") or "unknown-source").strip()
        title = str(row.get("title") or "untitled article").strip()
        label = f"{date_prefix} | {source} | {title}"
        option_map[label] = str(row["article_id"])
    return option_map


if st is not None:

    @st.cache_resource(show_spinner=False)
    def load_analysis_bundle(
        tickers: tuple[str, ...] | None,
        news_file: str,
        correlation_threshold: float,
        top_k: int | None,
        topic_weight: str,
    ) -> tuple[NetworkAnalyzer, dict[str, Any], dict[str, Any]]:
        """Load and cache the analyzer plus graph summary data."""

        selected_tickers = list(tickers) if tickers is not None else None
        # Cache the built analyzer so tab switches do not rerun the entire
        # local loading, news processing, and graph-building pipeline.
        analyzer, summary, context = build_analyzer_from_local_data(
            tickers=selected_tickers,
            news_file=news_file,
            correlation_threshold=correlation_threshold,
            top_k=top_k,
            topic_weight=topic_weight,
        )
        return analyzer, summary, context

    @st.cache_resource(show_spinner=False)
    def load_embedding_bundle(file_stem: str) -> dict[str, Any]:
        """Load and cache one article embedding bundle from disk."""

        store = LocalDataStore()
        return store.load_article_embedding_bundle(file_stem=file_stem)

else:

    def load_analysis_bundle(
        tickers: tuple[str, ...] | None,
        news_file: str,
        correlation_threshold: float,
        top_k: int | None,
        topic_weight: str,
    ) -> tuple[NetworkAnalyzer, dict[str, Any], dict[str, Any]]:
        """Raise a clear error when Streamlit is not installed."""

        raise RuntimeError(
            "Streamlit is not installed. Run 'pip install -r requirements.txt'."
        )

    def load_embedding_bundle(file_stem: str) -> dict[str, Any]:
        """Raise a clear error when Streamlit is not installed."""

        raise RuntimeError(
            "Streamlit is not installed. Run 'pip install -r requirements.txt'."
        )


def render_summary(summary: dict[str, Any], context: dict[str, Any]) -> None:
    """Render the graph summary metrics and count tables."""

    st.subheader("Graph Summary")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Price Tables", context["price_table_count"])
    metric_columns[1].metric("Sector Rows", context["sector_row_count"])
    metric_columns[2].metric("News Articles", context["article_count"])
    metric_columns[3].metric("Topic-Stock Rows", context["topic_stock_row_count"])

    node_frame = counter_to_frame(summary["node_counts"])
    edge_frame = counter_to_frame(summary["edge_counts"])
    left_column, right_column = st.columns(2)

    with left_column:
        st.caption("Node Counts")
        st.dataframe(node_frame, use_container_width=True, hide_index=True)

    with right_column:
        st.caption("Edge Counts")
        st.dataframe(edge_frame, use_container_width=True, hide_index=True)


def render_stock_tab(analyzer: NetworkAnalyzer, stock_options: list[str]) -> None:
    """Render the stock search view."""

    ticker = st.selectbox("Ticker", options=stock_options, key="stock_lookup_ticker")
    if not ticker:
        return

    try:
        stock_info = analyzer.get_stock_info(ticker)
    except NetworkAnalyzerError as exc:
        st.error(f"Stock lookup failed: {exc}")
        return

    st.markdown(f"### {stock_info['ticker']}")
    left_column, right_column = st.columns(2)
    with left_column:
        st.write(f"**Company:** {stock_info.get('company_name') or 'Unknown'}")
        st.write(f"**Sector:** {stock_info.get('sector') or 'Unknown'}")
    with right_column:
        st.write(f"**Industry:** {stock_info.get('industry') or 'Unknown'}")
        st.write(f"**Degree:** {stock_info.get('degree')}")
        st.write(
            f"**Degree Centrality:** {stock_info.get('degree_centrality', 0.0):.4f}"
        )

    st.caption("Top Stock Neighbors")
    st.dataframe(
        rows_to_frame(stock_info.get("top_neighbors", [])),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Related Topics")
    st.dataframe(
        rows_to_frame(stock_info.get("related_topics", [])),
        use_container_width=True,
        hide_index=True,
    )


def render_sector_tab(analyzer: NetworkAnalyzer, sector_options: list[str]) -> None:
    """Render the sector exploration view."""

    sector_name = st.selectbox(
        "Sector",
        options=sector_options,
        key="sector_lookup_name",
    )
    if not sector_name:
        return

    try:
        sector_info = analyzer.get_sector_info(sector_name)
    except NetworkAnalyzerError as exc:
        st.error(f"Sector lookup failed: {exc}")
        return

    st.markdown(f"### {sector_info['sector']}")
    st.write(f"**Stock Count:** {sector_info.get('stock_count')}")

    st.caption("Stocks")
    st.dataframe(
        rows_to_frame(sector_info.get("stocks", [])),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Connected Sectors")
    st.dataframe(
        rows_to_frame(sector_info.get("top_connected_sectors", [])),
        use_container_width=True,
        hide_index=True,
    )


def render_compare_tab(analyzer: NetworkAnalyzer, stock_options: list[str]) -> None:
    """Render the stock comparison view."""

    left_column, right_column = st.columns(2)
    with left_column:
        ticker_one = st.selectbox(
            "First Ticker",
            options=stock_options,
            key="compare_ticker_one",
        )
    with right_column:
        ticker_two = st.selectbox(
            "Second Ticker",
            options=stock_options,
            index=1 if len(stock_options) > 1 else 0,
            key="compare_ticker_two",
        )

    if not ticker_one or not ticker_two:
        return

    try:
        comparison = analyzer.compare_stocks(ticker_one, ticker_two)
    except NetworkAnalyzerError as exc:
        st.error(f"Stock comparison failed: {exc}")
        return

    st.write(f"**Same Sector:** {comparison.get('same_sector')}")
    st.write(f"**Degree Difference:** {comparison.get('degree_difference')}")
    st.write(
        f"**Common Topics:** {', '.join(comparison.get('common_topics', [])) or 'none'}"
    )

    direct_edge = comparison.get("direct_edge")
    if direct_edge:
        st.caption("Direct Edge")
        st.json(direct_edge)
    else:
        st.caption("Direct Edge")
        st.write("none")

    path_result = comparison.get("shortest_path", {})
    st.caption("Shortest Path")
    if path_result.get("path_found"):
        path_frame = pd.DataFrame(
            {
                "path_nodes": path_result.get("path_nodes", []),
            }
        )
        st.dataframe(path_frame, use_container_width=True, hide_index=True)
        st.write(f"**Edge Types:** {', '.join(path_result.get('edge_types', []))}")
    else:
        st.write("No path found.")


def render_path_tab(analyzer: NetworkAnalyzer) -> None:
    """Render the path finder view."""

    # Users pick readable labels in the UI, while the analyzer still receives
    # the exact node ids needed for graph traversal.
    node_option_map = build_node_option_map(analyzer)
    node_labels = list(node_option_map.keys())
    left_column, right_column = st.columns(2)

    with left_column:
        source_label = st.selectbox(
            "Start Node",
            options=node_labels,
            key="path_source_label",
        )
    with right_column:
        target_label = st.selectbox(
            "End Node",
            options=node_labels,
            index=1 if len(node_labels) > 1 else 0,
            key="path_target_label",
        )

    if not source_label or not target_label:
        return

    try:
        path_result = analyzer.find_shortest_path(
            node_option_map[source_label],
            node_option_map[target_label],
        )
    except NetworkAnalyzerError as exc:
        st.error(f"Path lookup failed: {exc}")
        return

    if not path_result.get("path_found"):
        st.write("No path found.")
        return

    st.write(f"**Path Length:** {path_result.get('path_length')}")
    st.dataframe(
        pd.DataFrame({"path_nodes": path_result.get("path_nodes", [])}),
        use_container_width=True,
        hide_index=True,
    )
    st.write(f"**Edge Types:** {', '.join(path_result.get('edge_types', []))}")


def render_centrality_tab(analyzer: NetworkAnalyzer) -> None:
    """Render the top-central-nodes view."""

    top_n = st.slider("Top N Nodes", min_value=5, max_value=25, value=10, step=5)
    centrality_metrics = analyzer.compute_centrality_metrics().head(top_n)
    st.dataframe(centrality_metrics, use_container_width=True, hide_index=True)


def render_llm_impact_tab(
    llm_tables: dict[str, pd.DataFrame] | None,
    selected_run: str | None,
) -> None:
    """Render sector and stock impact summaries from LLM-enriched news."""

    if llm_tables is None or selected_run is None:
        st.info(
            "No LLM impact file is loaded. Run `python enrich_news_with_llm.py ...` "
            "first, then select a run from the sidebar."
        )
        return

    summary_frame = llm_tables["article_llm_summary"]
    sector_frame = llm_tables["article_sector_impacts"]
    stock_frame = llm_tables["article_stock_impacts"]

    st.markdown(f"### LLM News Impact: `{selected_run}`")
    st.caption(
        "Scoring rule: magnitude = strength weight x confidence, where "
        "low=1, medium=2, high=3. Net score applies direction sign: "
        "positive=+1, negative=-1, mixed/uncertain=0."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Articles", len(summary_frame))
    metric_columns[1].metric("Sector Impact Rows", len(sector_frame))
    metric_columns[2].metric("Stock Impact Rows", len(stock_frame))
    metric_columns[3].metric(
        "Runs on Disk",
        "loaded",
    )

    sector_summary = summarize_llm_impacts(sector_frame, entity_column="sector")
    stock_summary = summarize_llm_impacts(stock_frame, entity_column="ticker")
    article_detail = build_article_impact_detail(summary_frame, sector_frame, stock_frame)

    if not sector_summary.empty:
        st.caption("Sector Impact Summary")
        st.bar_chart(
            sector_summary.set_index("sector")[
                ["gross_impact_score", "net_impact_score"]
            ]
        )
        st.dataframe(sector_summary, use_container_width=True, hide_index=True)
    else:
        st.caption("Sector Impact Summary")
        st.write("No sector impact rows for the current selection.")

    if not stock_summary.empty:
        st.caption("Stock Impact Summary")
        st.bar_chart(
            stock_summary.set_index("ticker")[
                ["gross_impact_score", "net_impact_score"]
            ]
        )
        st.dataframe(stock_summary, use_container_width=True, hide_index=True)
    else:
        st.caption("Stock Impact Summary")
        st.write("No stock impact rows for the current selection.")

    st.caption("Article Impact Detail")
    st.dataframe(article_detail, use_container_width=True, hide_index=True)


def render_semantic_news_tab(
    embedding_bundle: dict[str, Any] | None,
    selected_run: str | None,
    tickers: list[str] | None,
) -> None:
    """Render semantic nearest-neighbor retrieval over embedded articles."""

    if SemanticNewsRetriever is None:
        st.info(
            "Semantic retrieval helpers are not available in this bundle. "
            "The rest of the query app can still run normally."
        )
        return

    if embedding_bundle is None or selected_run is None:
        st.info(
            "No embedding run is loaded. Run `python article_embeddings.py ...` "
            "first, then select a run from the sidebar."
        )
        return

    filtered_bundle = filter_embedding_bundle_by_tickers(embedding_bundle, tickers)
    metadata = filtered_bundle["metadata"]
    vectors = filtered_bundle["vectors"]
    if metadata.empty:
        st.info("No embedded articles match the current ticker filter.")
        return

    try:
        retriever = SemanticNewsRetriever(metadata=metadata, vectors=vectors)
    except SemanticRetrieverError as exc:
        st.error(f"Semantic retrieval could not start: {exc}")
        return

    provider = str(metadata["embedding_provider"].iloc[0])
    model = str(metadata["embedding_model"].iloc[0])
    vector_dim = int(metadata["vector_dim"].iloc[0])

    st.markdown(f"### Semantic News Retrieval: `{selected_run}`")
    st.caption(
        "This view retrieves nearest-neighbor articles in embedding space. "
        "Use a keyword to narrow the source article list, then inspect the top matches."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Embedded Articles", len(metadata))
    metric_columns[1].metric("Vector Dim", vector_dim)
    metric_columns[2].metric("Provider", provider)
    metric_columns[3].metric("Model", model)

    keyword = st.text_input(
        "Keyword Filter",
        value="",
        help="Optional title/summary keyword filter for the source-article picker.",
        key="semantic_keyword_filter",
    )
    top_k = st.slider(
        "Top Similar Articles",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="semantic_top_k",
    )

    candidate_frame = filter_semantic_candidates(metadata, keyword=keyword, limit=200)
    if candidate_frame.empty:
        st.warning("No source articles matched the current keyword filter.")
        return

    option_map = build_article_option_map(candidate_frame)
    selected_label = st.selectbox(
        "Source Article",
        options=list(option_map.keys()),
        key="semantic_source_article",
    )
    if not selected_label:
        return

    source_article = retriever.get_article(option_map[selected_label])
    similar_articles = retriever.find_similar_articles(
        article_id=option_map[selected_label],
        top_k=top_k,
    )
    result_frame = similar_articles.copy()
    if not result_frame.empty:
        result_frame["similarity_score"] = result_frame["similarity_score"].round(4)
        result_frame["summary"] = result_frame["summary"].fillna("").str.slice(0, 220)

    st.caption("Seed Article")
    left_column, right_column = st.columns(2)
    with left_column:
        st.write(f"**Title:** {source_article.get('title') or 'Unknown'}")
        st.write(f"**Source:** {source_article.get('source') or 'Unknown'}")
    with right_column:
        st.write(f"**Published At:** {source_article.get('published_at')}")
        st.write(f"**Tickers:** {source_article.get('tickers') or 'None'}")

    st.write(source_article.get("summary") or "No summary available.")

    st.caption("Most Similar Articles")
    st.dataframe(result_frame, use_container_width=True, hide_index=True)


def render_new_news_analysis_tab(analyzer: NetworkAnalyzer) -> None:
    """Render one-shot LLM assessment for a new user-provided news item."""

    st.markdown("### New News Analysis")
    st.caption(
        "Enter a new headline and summary. The system will assess likely impacted "
        "topics, sectors, and stocks inside the current graph universe, then show "
        "how those results map onto the existing network."
    )

    universe = extract_graph_universe(analyzer)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Allowed Stocks", len(universe["tickers"]))
    metric_columns[1].metric("Allowed Sectors", len(universe["sectors"]))
    metric_columns[2].metric("Allowed Topics", len(universe["topics"]))

    form_key = "new_news_analysis_form"
    state_key = "new_news_assessment_payload"
    with st.form(form_key):
        title = st.text_input(
            "News Title",
            value="",
            placeholder="Example: Apple announces major enterprise AI partnership",
        )
        summary = st.text_area(
            "News Summary",
            value="",
            height=180,
            placeholder=(
                "Paste a short summary of a new article. The model will not predict "
                "prices. It will estimate likely topic, sector, and stock impacts."
            ),
        )
        submitted = st.form_submit_button("Analyze New News")

    if submitted:
        if not title.strip() and not summary.strip():
            st.warning("Enter at least a title or a summary before running analysis.")
        else:
            try:
                with st.spinner("Running LLM impact assessment..."):
                    llm_analyzer = LLMNewsImpactAnalyzer()
                    assessment = llm_analyzer.assess_news(
                        title=title,
                        summary=summary,
                        allowed_topics=universe["topics"],
                        allowed_sectors=universe["sectors"],
                        allowed_tickers=universe["tickers"],
                    )
                st.session_state[state_key] = assessment.model_dump()
            except LLMNewsImpactAnalyzerError as exc:
                st.error(f"New news analysis failed: {exc}")
                return

    raw_assessment = st.session_state.get(state_key)
    if raw_assessment is None:
        st.info(
            "Run one analysis to see impacted sectors, stocks, topics, and graph-grounded context."
        )
        return

    assessment = NewNewsImpactAssessment.model_validate(raw_assessment)
    assessment_frames = build_assessment_frames(assessment)
    grounding_frames = build_graph_grounding_frames(assessment, analyzer)
    graph_augmenter = NewsGraphAugmenter()
    preview = graph_augmenter.build_preview(assessment, analyzer.graph)
    preview_summary = preview["summary"]

    st.write(f"**Event Summary:** {assessment.event_summary}")
    st.write(f"**Primary Event Type:** {assessment.primary_event_type}")
    st.write(f"**Overall Market Relevance:** {assessment.overall_market_relevance}")
    st.write(f"**Overall Rationale:** {assessment.overall_rationale}")

    st.caption("Assessed Topics")
    st.dataframe(assessment_frames["topics"], use_container_width=True, hide_index=True)

    st.caption("Assessed Sector Impacts")
    st.dataframe(
        assessment_frames["sector_impacts"],
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Assessed Stock Impacts")
    st.dataframe(
        assessment_frames["stock_impacts"],
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Graph-Grounded Topic Context")
    st.dataframe(
        grounding_frames["topic_grounding"],
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Graph-Grounded Sector Context")
    st.dataframe(
        grounding_frames["sector_grounding"],
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Graph-Grounded Stock Context")
    st.dataframe(
        grounding_frames["stock_grounding"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Graph Augmentation Preview")
    st.caption(
        "The system converts this one new-news assessment into synthetic "
        "`topic_stock` rows, overlays them on the current graph, and previews "
        "which stock-topic relationships would be added or strengthened. This is "
        "a structure preview, not a price forecast."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Synthetic Rows", preview_summary["synthetic_row_count"])
    metric_columns[1].metric("New Stock-Topic Edges", preview_summary["new_edges"])
    metric_columns[2].metric(
        "Strengthened Edges",
        preview_summary["strengthened_edges"],
    )
    metric_columns[3].metric(
        "Total Preview Impact",
        f"{preview_summary['total_impact_score']:.2f}",
    )

    detail_columns = st.columns(4)
    detail_columns[0].metric("Impacted Stocks", preview_summary["impacted_stocks"])
    detail_columns[1].metric("Impacted Sectors", preview_summary["impacted_sectors"])
    detail_columns[2].metric("Impacted Topics", preview_summary["impacted_topics"])
    detail_columns[3].metric(
        "Stock-Topic Edge Count",
        (
            f"{preview_summary['base_stock_topic_edges']} -> "
            f"{preview_summary['augmented_stock_topic_edges']}"
        ),
    )

    synthetic_frame = preview["synthetic_topic_stock"].copy()
    if not synthetic_frame.empty:
        st.caption("Synthetic topic_stock Rows")
        st.dataframe(
            synthetic_frame[
                [
                    "ticker",
                    "topic",
                    "sector",
                    "impact_direction",
                    "impact_strength",
                    "impact_score",
                    "avg_topic_relevance",
                    "avg_ticker_relevance",
                    "avg_ticker_sentiment",
                    "avg_overall_sentiment",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    update_frame = preview["stock_topic_updates"].copy()
    if not update_frame.empty:
        st.caption("Projected Stock-Topic Edge Updates")
        st.dataframe(
            update_frame[
                [
                    "ticker",
                    "topic",
                    "sector",
                    "edge_status",
                    "existing_article_count",
                    "projected_article_count",
                    "impact_score",
                    "stock_confidence",
                    "topic_confidence",
                    "sector_support",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.caption("Projected Sector Exposure")
    st.dataframe(
        preview["sector_exposure"],
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Projected Topic Exposure")
    st.dataframe(
        preview["topic_exposure"],
        use_container_width=True,
        hide_index=True,
    )


def run_app(*, configure_page: bool = True, embedded: bool = False) -> None:
    """Run the Streamlit application."""

    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Run 'pip install -r requirements.txt'."
        )

    if configure_page:
        st.set_page_config(page_title="Market Query Explorer", layout="wide")
    inject_query_app_styles()
    st.title("Market Query Explorer")
    st.caption(
        "Query-focused market dashboard built on the cached local graph. "
        "Use this site for overview, lookups, comparisons, and graph-grounded news analysis."
    )

    with st.sidebar:
        st.header("Configuration")
        render_sidebar_course_guide()
        ticker_text = st.text_input(
            "Ticker Filter",
            value=",".join(DEFAULT_WEB_TICKERS),
            help=(
                "Optional comma-separated tickers to limit the local graph. "
                "The default uses a broader multi-sector core set."
            ),
        )
        news_file = st.text_input(
            "News File",
            value="merged_seed_news.json",
            help="File name inside data/raw/news/.",
        )
        correlation_threshold = st.slider(
            "Correlation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )
        top_k_value = st.number_input(
            "Top-K Stock Neighbors",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
            help="Set to 0 to disable top-k mode and use threshold mode only.",
        )
        topic_weight = st.selectbox(
            "Stock-Topic Weight",
            options=VALID_TOPIC_WEIGHTS,
            index=0,
        )
        st.header("LLM Impact")
        llm_store = LocalDataStore()
        llm_run_options = llm_store.list_llm_impact_runs()
        selected_llm_run = st.selectbox(
            "LLM Impact Run",
            options=["None"] + llm_run_options,
            index=0 if not llm_run_options else 1,
            help=(
                "Optional offline LLM enrichment run from data/raw/llm_enriched/. "
                "Choose None to hide the impact tab content."
            ),
        )
        include_semantic = semantic_features_available()
        selected_embedding_run = "None"
        if include_semantic:
            st.header("Semantic Retrieval")
            embedding_run_options = llm_store.list_embedding_runs()
            selected_embedding_run = st.selectbox(
                "Embedding Run",
                options=["None"] + embedding_run_options,
                index=0 if not embedding_run_options else 1,
                help=(
                    "Optional offline article embedding run from data/raw/embeddings/. "
                    "Choose None to hide the semantic retrieval tab content."
                ),
            )
        else:
            st.caption(
                "Semantic retrieval is not included in this course-only bundle. "
                "The core graph and LLM impact features are fully available."
            )

    if embedded:
        st.caption(
            "Use the sidebar workspace switch to move between the analysis dashboard "
            "and the interactive graph explorer."
        )
    else:
        st.info(
            "For the draggable network view, launch the separate site with "
            "`streamlit run interactive_graph_app.py`."
        )

    tickers = parse_ticker_text(ticker_text)
    top_k = normalize_top_k(int(top_k_value))

    try:
        with st.spinner("Building local graph..."):
            # The web query page reuses the same local graph assembly path as
            # the CLI so both entrypoints stay consistent.
            analyzer, summary, context = load_analysis_bundle(
                tickers=tuple(tickers) if tickers is not None else None,
                news_file=news_file,
                correlation_threshold=correlation_threshold,
                top_k=top_k,
                topic_weight=topic_weight,
            )
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    render_market_overview(analyzer, summary, context)

    with st.expander("Graph Summary Tables", expanded=False):
        render_summary(summary, context)

    with st.expander("Current Local Configuration", expanded=False):
        st.write(
            {
                "tickers": tickers,
                "news_file": news_file,
                "correlation_threshold": correlation_threshold,
                "top_k": top_k,
                "topic_weight": topic_weight,
            }
        )

    stock_options = sorted(
        [
            node_data.get("ticker")
            for _, node_data in analyzer.graph.nodes(data=True)
            if node_data.get("node_type") == "stock" and node_data.get("ticker")
        ]
    )
    sector_options = sorted(
        [
            node_data.get("sector")
            for _, node_data in analyzer.graph.nodes(data=True)
            if node_data.get("node_type") == "sector" and node_data.get("sector")
        ]
    )

    llm_tables: dict[str, pd.DataFrame] | None = None
    if selected_llm_run != "None":
        try:
            raw_llm_tables = llm_store.load_llm_impact_tables(selected_llm_run)
            llm_tables = filter_llm_impact_tables(
                raw_llm_tables,
                tickers=tickers,
                sector_options=sector_options,
            )
        except LocalDataStoreError as exc:
            st.warning(f"LLM impact data could not be loaded: {exc}")

    embedding_bundle: dict[str, Any] | None = None
    if include_semantic and selected_embedding_run != "None":
        try:
            embedding_bundle = load_embedding_bundle(selected_embedding_run)
        except LocalDataStoreError as exc:
            st.warning(f"Embedding data could not be loaded: {exc}")

    tab_labels = build_query_tab_labels(include_semantic=include_semantic)
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_stock_tab(analyzer, stock_options)
    with tabs[1]:
        render_sector_tab(analyzer, sector_options)
    with tabs[2]:
        render_compare_tab(analyzer, stock_options)
    with tabs[3]:
        render_path_tab(analyzer)
    with tabs[4]:
        render_centrality_tab(analyzer)
    with tabs[5]:
        render_llm_impact_tab(
            llm_tables=llm_tables,
            selected_run=None if selected_llm_run == "None" else selected_llm_run,
        )
    with tabs[6]:
        render_new_news_analysis_tab(analyzer)
    if include_semantic:
        with tabs[7]:
            render_semantic_news_tab(
                embedding_bundle=embedding_bundle,
                selected_run=(
                    None if selected_embedding_run == "None" else selected_embedding_run
                ),
                tickers=tickers,
            )


if __name__ == "__main__":
    run_app()
