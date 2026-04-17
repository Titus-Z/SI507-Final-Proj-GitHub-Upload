"""Streamlit app for draggable graph visualization of the market network."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from streamlit_app import (
    DEFAULT_WEB_TICKERS,
    load_analysis_bundle,
    normalize_top_k,
    parse_ticker_text,
)

try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:  # pragma: no cover - handled at runtime
    st = None
    components = None

try:
    from pyvis.network import Network
except ImportError:  # pragma: no cover - handled at runtime
    Network = None


NODE_COLORS = {
    "stock": "#2F6BFF",
    "sector": "#1E9E63",
    "topic": "#F08C2E",
    "unknown": "#7F8C8D",
}

GRAPH_CARD_COLORS = [
    "#2563EB",
    "#0F766E",
    "#D97706",
    "#9333EA",
]


def node_color(node_type: str) -> str:
    """Return a stable color for one node type."""

    return NODE_COLORS.get(node_type, NODE_COLORS["unknown"])


def node_size(node_data: dict[str, Any], degree: int) -> int:
    """Compute a readable node size from type and light metadata."""

    node_type = node_data.get("node_type")
    if node_type == "sector":
        return 26 + min(degree * 2, 12)
    if node_type == "stock":
        return 18 + min(degree * 2, 14)
    if node_type == "topic":
        article_count = int(node_data.get("article_count") or 0)
        return 12 + min(article_count // 10, 16)
    return 16


def edge_width(edge_data: dict[str, Any]) -> float:
    """Map edge metadata to a visible line width."""

    edge_type = edge_data.get("edge_type")
    if edge_type == "stock_stock":
        correlation = float(edge_data.get("correlation") or 0.0)
        return max(1.0, round(correlation * 6, 2))
    if edge_type == "stock_topic":
        article_count = int(edge_data.get("article_count") or 0)
        return max(1.0, min(6.0, 1.0 + article_count / 20))
    if edge_type == "sector_sector":
        average_correlation = float(edge_data.get("average_correlation") or 0.0)
        return max(1.0, round(average_correlation * 6, 2))
    return 1.5


def build_node_title(node_id: str, node_data: dict[str, Any], degree: int) -> str:
    """Build tooltip text for one node."""

    node_type = node_data.get("node_type", "unknown")
    lines = [f"id: {node_id}", f"type: {node_type}", f"label: {node_data.get('label')}"]

    if node_type == "stock":
        lines.extend(
            [
                f"ticker: {node_data.get('ticker')}",
                f"company: {node_data.get('company_name') or 'Unknown'}",
                f"sector: {node_data.get('sector') or 'Unknown'}",
                f"industry: {node_data.get('industry') or 'Unknown'}",
            ]
        )
    elif node_type == "sector":
        lines.extend(
            [
                f"sector: {node_data.get('sector')}",
                f"stock_count: {node_data.get('stock_count')}",
            ]
        )
    elif node_type == "topic":
        lines.extend(
            [
                f"topic: {node_data.get('topic')}",
                f"article_count: {node_data.get('article_count')}",
            ]
        )

    lines.append(f"degree: {degree}")
    return "<br>".join(lines)


def build_edge_title(edge_data: dict[str, Any]) -> str:
    """Build tooltip text for one edge."""

    edge_type = edge_data.get("edge_type", "unknown")
    lines = [f"edge_type: {edge_type}"]

    if edge_type == "stock_stock":
        lines.extend(
            [
                f"correlation: {edge_data.get('correlation')}",
                f"overlap_days: {edge_data.get('overlap_days')}",
            ]
        )
    elif edge_type == "stock_topic":
        lines.extend(
            [
                f"weight: {edge_data.get('weight')}",
                f"article_count: {edge_data.get('article_count')}",
                f"avg_ticker_sentiment: {edge_data.get('avg_ticker_sentiment')}",
            ]
        )
    elif edge_type == "sector_sector":
        lines.extend(
            [
                f"average_correlation: {edge_data.get('average_correlation')}",
                f"stock_edge_count: {edge_data.get('stock_edge_count')}",
            ]
        )
    else:
        lines.append(f"weight: {edge_data.get('weight')}")

    return "<br>".join(lines)


def filter_graph(
    graph: nx.Graph,
    included_node_types: set[str],
    included_edge_types: set[str],
    remove_isolates: bool,
) -> nx.Graph:
    """Filter the graph by node and edge type selections."""

    filtered_graph = graph.copy()

    # Remove node types first so subsequent edge filtering only works with the
    # surviving part of the graph.
    node_ids_to_remove = [
        node_id
        for node_id, node_data in filtered_graph.nodes(data=True)
        if node_data.get("node_type") not in included_node_types
    ]
    filtered_graph.remove_nodes_from(node_ids_to_remove)

    edge_ids_to_remove = [
        (source, target)
        for source, target, edge_data in filtered_graph.edges(data=True)
        if edge_data.get("edge_type") not in included_edge_types
    ]
    filtered_graph.remove_edges_from(edge_ids_to_remove)

    if remove_isolates:
        isolate_nodes = list(nx.isolates(filtered_graph))
        filtered_graph.remove_nodes_from(isolate_nodes)

    return filtered_graph


def build_pyvis_html(
    graph: nx.Graph,
    physics_enabled: bool,
    show_physics_controls: bool,
    height_px: int,
) -> str:
    """Convert a NetworkX graph into draggable Pyvis HTML."""

    if Network is None:
        raise RuntimeError(
            "Pyvis is not installed. Run 'pip install -r requirements.txt'."
        )

    network = Network(
        height=f"{height_px}px",
        width="100%",
        directed=False,
        notebook=False,
        neighborhood_highlight=True,
        select_menu=True,
        filter_menu=True,
        bgcolor="#FFFFFF",
        font_color="#111111",
        cdn_resources="in_line",
    )

    # Translate the NetworkX graph into Pyvis nodes so the browser can render a
    # draggable physics-based network.
    for node_id, node_data in graph.nodes(data=True):
        degree = graph.degree(node_id)
        network.add_node(
            node_id,
            label=str(node_data.get("label") or node_id),
            title=build_node_title(node_id, node_data, degree),
            color=node_color(str(node_data.get("node_type", "unknown"))),
            size=node_size(node_data, degree),
        )

    for source, target, edge_data in graph.edges(data=True):
        network.add_edge(
            source,
            target,
            title=build_edge_title(edge_data),
            value=edge_width(edge_data),
            width=edge_width(edge_data),
            color="#B0B8C5",
        )

    # Use the built-in physics helpers instead of overriding the full options
    # object, which keeps the optional control panel compatible with Pyvis.
    network.barnes_hut()
    network.toggle_physics(physics_enabled)
    network.options.interaction.hover = True
    network.options.interaction.navigationButtons = True
    network.options.interaction.keyboard = True
    network.options.edges.smooth = False

    if show_physics_controls:
        network.show_buttons(filter_=["physics"])

    return network.generate_html(notebook=False)


def counter_to_frame(counter_data) -> pd.DataFrame:
    """Convert counter-like objects into display tables."""

    rows = [
        {"type": item_type, "count": count}
        for item_type, count in sorted(counter_data.items())
    ]
    return pd.DataFrame(rows)


def inject_graph_app_styles() -> None:
    """Apply a lightweight explorer theme to the draggable graph app."""

    if st is None:
        return

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 24%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.10), transparent 26%),
                linear-gradient(180deg, #F7F9FC 0%, #EEF2F7 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0F172A 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        section[data-testid="stSidebar"] * {
            color: #E5EEF8;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.90);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }
        .graph-hero {
            padding: 1.3rem 1.45rem;
            border-radius: 24px;
            color: white;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(37, 99, 235, 0.94) 100%);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.16);
            margin-bottom: 1rem;
        }
        .graph-hero h2 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
        }
        .graph-hero p {
            margin: 0;
            color: rgba(255, 255, 255, 0.92);
            line-height: 1.55;
        }
        .graph-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .graph-card h3 {
            margin-top: 0;
            margin-bottom: 0.25rem;
            font-size: 1.05rem;
        }
        .legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.75rem;
        }
        .legend-card {
            background: rgba(255, 255, 255, 0.94);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
        .legend-chip {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            margin-right: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_graph_type_frame(graph: nx.Graph, item: str) -> pd.DataFrame:
    """Summarize filtered node or edge counts by type for display panels."""

    if item == "node":
        counts: dict[str, int] = {}
        for _, node_data in graph.nodes(data=True):
            node_type = str(node_data.get("node_type", "unknown"))
            counts[node_type] = counts.get(node_type, 0) + 1
        return counter_to_frame(counts)

    counts = {}
    for _, _, edge_data in graph.edges(data=True):
        edge_type = str(edge_data.get("edge_type", "unknown"))
        counts[edge_type] = counts.get(edge_type, 0) + 1
    return counter_to_frame(counts)


def build_top_degree_frame(graph: nx.Graph, limit: int = 8) -> pd.DataFrame:
    """Build one small degree leaderboard for the filtered graph."""

    rows: list[dict[str, Any]] = []
    for node_id, node_data in graph.nodes(data=True):
        rows.append(
            {
                "label": node_data.get("label") or node_id,
                "node_type": node_data.get("node_type", "unknown"),
                "degree": graph.degree(node_id),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["label", "node_type", "degree"])

    return (
        pd.DataFrame(rows)
        .sort_values(["degree", "node_type", "label"], ascending=[False, True, True])
        .head(limit)
        .reset_index(drop=True)
    )


def run_app() -> None:
    """Run the interactive graph visualization app."""

    if st is None or components is None:
        raise RuntimeError(
            "Streamlit is not installed. Run 'pip install -r requirements.txt'."
        )

    st.set_page_config(page_title="Interactive Market Graph", layout="wide")
    inject_graph_app_styles()
    st.title("Interactive Market Graph Explorer")
    st.caption(
        "Draggable graph explorer built from the local project data. "
        "Use this page to inspect structure, clusters, and bridge nodes."
    )

    with st.sidebar:
        st.header("Graph Input")
        ticker_text = st.text_input(
            "Ticker Filter",
            value=",".join(DEFAULT_WEB_TICKERS),
            help=(
                "Optional comma-separated tickers to limit the local graph. "
                "The default uses a broader multi-sector core set."
            ),
        )
        news_file = st.text_input("News File", value="merged_seed_news.json")
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
            help="Set to 0 to disable top-k mode.",
        )
        topic_weight = st.selectbox(
            "Stock-Topic Weight",
            options=[
                "article_count",
                "avg_topic_relevance",
                "avg_ticker_relevance",
                "avg_ticker_sentiment",
                "avg_overall_sentiment",
            ],
            index=0,
        )

        st.header("Graph Display")
        included_node_types = set(
            st.multiselect(
                "Node Types",
                options=["stock", "sector", "topic"],
                default=["stock", "sector", "topic"],
            )
        )
        included_edge_types = set(
            st.multiselect(
                "Edge Types",
                options=["stock_stock", "stock_sector", "stock_topic", "sector_sector"],
                default=["stock_stock", "stock_sector", "stock_topic", "sector_sector"],
            )
        )
        remove_isolates = st.checkbox("Remove Isolated Nodes", value=True)
        physics_enabled = st.checkbox("Enable Physics", value=True)
        show_physics_controls = st.checkbox("Show Physics Controls", value=True)
        height_px = st.slider("Graph Height", min_value=500, max_value=1100, value=760)

    st.info(
        "For structured stock / sector / path queries, use the separate site with "
        "`streamlit run query_app.py`."
    )

    tickers = parse_ticker_text(ticker_text)
    top_k = normalize_top_k(int(top_k_value))

    if not included_node_types:
        st.error("Select at least one node type.")
        st.stop()
    if not included_edge_types:
        st.error("Select at least one edge type.")
        st.stop()

    try:
        with st.spinner("Building local graph..."):
            # Keep this app on the exact same graph-building pipeline as the
            # query app; only the presentation layer changes here.
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

    # Filtering happens before HTML generation so the visualization layer only
    # receives the subset the user actually wants to inspect.
    filtered_graph = filter_graph(
        analyzer.graph,
        included_node_types=included_node_types,
        included_edge_types=included_edge_types,
        remove_isolates=remove_isolates,
    )

    st.markdown(
        """
        <div class="graph-hero">
            <h2>Interactive Structure Browser</h2>
            <p>
                This page is tuned for visual structure first. It follows the same
                graph-building pipeline as the query dashboard, but packages the output
                like a graph browser: fast summary, clear legend, then the draggable network.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Price Tables", context["price_table_count"])
    metric_columns[1].metric("Articles", context["article_count"])
    metric_columns[2].metric("Filtered Nodes", filtered_graph.number_of_nodes())
    metric_columns[3].metric("Filtered Edges", filtered_graph.number_of_edges())
    metric_columns[4].metric("Topics", summary["node_counts"].get("topic", 0))

    node_type_frame = build_graph_type_frame(filtered_graph, item="node")
    edge_type_frame = build_graph_type_frame(filtered_graph, item="edge")
    top_degree_frame = build_top_degree_frame(filtered_graph)

    details_column, legend_column = st.columns([3, 2])
    with details_column:
        st.markdown(
            """
            <div class="graph-card">
                <h3>Filtered Graph Composition</h3>
                <p>
                    This panel shows what remains after the current node and edge filters.
                    It is useful for checking whether the visual graph still reflects the slice
                    of the market you intended to study.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        left_inner, right_inner = st.columns(2)
        with left_inner:
            st.caption("Filtered Node Types")
            st.dataframe(node_type_frame, use_container_width=True, hide_index=True)
        with right_inner:
            st.caption("Filtered Edge Types")
            st.dataframe(edge_type_frame, use_container_width=True, hide_index=True)

        st.caption("Top Nodes by Degree")
        st.dataframe(top_degree_frame, use_container_width=True, hide_index=True)

    with legend_column:
        st.markdown(
            """
            <div class="graph-card">
                <h3>Legend And Controls</h3>
                <p>
                    Color stays fixed by node type so you can quickly separate companies,
                    sectors, and themes while dragging the graph.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        legend_html = ['<div class="legend-grid">']
        for index, node_type in enumerate(["stock", "sector", "topic"]):
            legend_html.append(
                f"""
                <div class="legend-card" style="border-top: 4px solid {GRAPH_CARD_COLORS[index]};">
                    <div><span class="legend-chip" style="background:{NODE_COLORS[node_type]};"></span>{node_type}</div>
                    <div style="margin-top:0.45rem;color:#475569;font-size:0.9rem;">
                        color code for {node_type} nodes
                    </div>
                </div>
                """
            )
        legend_html.append("</div>")
        st.markdown("".join(legend_html), unsafe_allow_html=True)
        st.caption("Display Controls")
        st.write(
            {
                "remove_isolates": remove_isolates,
                "physics_enabled": physics_enabled,
                "show_physics_controls": show_physics_controls,
                "height_px": height_px,
            }
        )

    if filtered_graph.number_of_nodes() == 0:
        st.warning("No nodes remain after filtering. Adjust the sidebar filters.")
        return

    html = build_pyvis_html(
        filtered_graph,
        physics_enabled=physics_enabled,
        show_physics_controls=show_physics_controls,
        height_px=height_px,
    )
    components.html(html, height=height_px + 50, scrolling=True)


if __name__ == "__main__":
    run_app()
