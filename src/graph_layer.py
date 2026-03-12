"""
Graph Processing Layer for InvestIQ DataScience.

This module converts normalized entity modules into graph objects and computes
core topological metrics for the Indian deeptech investor network.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import pandas as pd


def investor_node_id(investor_id: int) -> str:
    return f"investor_{investor_id}"


def startup_node_id(startup_id: int) -> str:
    return f"startup_{startup_id}"


def build_bipartite_graph(
    startups: pd.DataFrame,
    investors: pd.DataFrame,
    transactions: pd.DataFrame,
) -> nx.Graph:
    """Build an investor↔startup bipartite graph from entity modules."""
    graph = nx.Graph()

    for _, row in investors.iterrows():
        node = investor_node_id(int(row["investor_id"]))
        graph.add_node(
            node,
            bipartite=0,
            entity_type="investor",
            investor_id=int(row["investor_id"]),
            name=row["InvestorsName"],
            investor_type=row.get("InvestorType", "Unknown"),
        )

    for _, row in startups.iterrows():
        node = startup_node_id(int(row["startup_id"]))
        graph.add_node(
            node,
            bipartite=1,
            entity_type="startup",
            startup_id=int(row["startup_id"]),
            name=row["StartupName"],
            industry=row.get("IndustryVertical"),
            subvertical=row.get("SubVertical"),
            city=row.get("City"),
        )

    for _, row in transactions.iterrows():
        investor_node = investor_node_id(int(row["investor_id"]))
        startup_node = startup_node_id(int(row["startup_id"]))
        graph.add_edge(
            investor_node,
            startup_node,
            edge_id=int(row["edge_id"]),
            date=row.get("Date"),
            amount=row.get("AmountInUSD"),
            funding_stage=row.get("InvestmentType"),
        )

    return graph


def get_investor_nodes(graph: nx.Graph) -> list[str]:
    return [node for node, data in graph.nodes(data=True) if data.get("bipartite") == 0]


def get_startup_nodes(graph: nx.Graph) -> list[str]:
    return [node for node, data in graph.nodes(data=True) if data.get("bipartite") == 1]


def project_investor_graph(graph: nx.Graph) -> nx.Graph:
    """Project bipartite graph to weighted investor co-investment graph."""
    investor_nodes = get_investor_nodes(graph)
    return nx.bipartite.weighted_projected_graph(graph, investor_nodes)


def project_startup_graph(graph: nx.Graph) -> nx.Graph:
    """Project bipartite graph to weighted startup shared-investor graph."""
    startup_nodes = get_startup_nodes(graph)
    return nx.bipartite.weighted_projected_graph(graph, startup_nodes)


def _safe_eigenvector_centrality(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:
        return {node: 0.0 for node in graph.nodes()}


def _safe_pagerank(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(graph, weight="weight")
    except Exception:
        return {node: 0.0 for node in graph.nodes()}


def _safe_betweenness(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    return nx.betweenness_centrality(graph, weight="weight", normalized=True)


def _safe_louvain_communities(graph: nx.Graph) -> list[set[str]]:
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [{node} for node in graph.nodes()]
    try:
        return nx.community.louvain_communities(graph, weight="weight", seed=42)
    except Exception:
        return [set(component) for component in nx.connected_components(graph)]


def _community_lookup(communities: Iterable[set[str]]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            lookup[node] = community_id
    return lookup


def compute_investor_metrics(projected_investor_graph: nx.Graph) -> pd.DataFrame:
    """Compute node-level topological metrics on investor co-investment graph."""
    if projected_investor_graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "node",
                "degree",
                "weighted_degree",
                "degree_centrality",
                "eigenvector_centrality",
                "pagerank",
                "betweenness_centrality",
                "community_id",
            ]
        )

    degree = dict(projected_investor_graph.degree())
    weighted_degree = dict(projected_investor_graph.degree(weight="weight"))
    degree_centrality = nx.degree_centrality(projected_investor_graph)
    eigenvector_centrality = _safe_eigenvector_centrality(projected_investor_graph)
    pagerank = _safe_pagerank(projected_investor_graph)
    betweenness = _safe_betweenness(projected_investor_graph)

    communities = _safe_louvain_communities(projected_investor_graph)
    community_map = _community_lookup(communities)

    records = []
    for node, attrs in projected_investor_graph.nodes(data=True):
        records.append(
            {
                "node": node,
                "investor_id": attrs.get("investor_id"),
                "name": attrs.get("name"),
                "investor_type": attrs.get("investor_type"),
                "degree": degree.get(node, 0),
                "weighted_degree": weighted_degree.get(node, 0.0),
                "degree_centrality": degree_centrality.get(node, 0.0),
                "eigenvector_centrality": eigenvector_centrality.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "community_id": community_map.get(node, -1),
            }
        )

    return pd.DataFrame(records).sort_values(
        by=["pagerank", "weighted_degree"], ascending=False
    )


def compute_startup_metrics(graph: nx.Graph) -> pd.DataFrame:
    """Compute startup-side degree metrics directly on the bipartite graph."""
    startup_nodes = get_startup_nodes(graph)
    if not startup_nodes:
        return pd.DataFrame(columns=["node", "degree", "degree_centrality"])

    degree = dict(graph.degree(startup_nodes))
    degree_centrality = nx.degree_centrality(graph)

    records = []
    for node in startup_nodes:
        attrs = graph.nodes[node]
        records.append(
            {
                "node": node,
                "startup_id": attrs.get("startup_id"),
                "name": attrs.get("name"),
                "industry": attrs.get("industry"),
                "subvertical": attrs.get("subvertical"),
                "city": attrs.get("city"),
                "degree": degree.get(node, 0),
                "degree_centrality": degree_centrality.get(node, 0.0),
            }
        )

    return pd.DataFrame(records).sort_values(by=["degree", "degree_centrality"], ascending=False)


def summarize_graph(graph: nx.Graph, projected_investor_graph: nx.Graph | None = None) -> dict[str, float | int]:
    """Return macro-level graph summary statistics."""
    projected_investor_graph = projected_investor_graph or project_investor_graph(graph)

    investor_nodes = get_investor_nodes(graph)
    startup_nodes = get_startup_nodes(graph)

    summary: dict[str, float | int] = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_investors": len(investor_nodes),
        "num_startups": len(startup_nodes),
        "num_connected_components": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
        "bipartite_density": nx.density(graph) if graph.number_of_nodes() else 0.0,
        "investor_projection_nodes": projected_investor_graph.number_of_nodes(),
        "investor_projection_edges": projected_investor_graph.number_of_edges(),
        "investor_projection_density": nx.density(projected_investor_graph) if projected_investor_graph.number_of_nodes() else 0.0,
    }

    if projected_investor_graph.number_of_nodes() > 0 and projected_investor_graph.number_of_edges() > 0:
        communities = _safe_louvain_communities(projected_investor_graph)
        summary["investor_communities"] = len(communities)
        try:
            summary["louvain_modularity"] = nx.community.modularity(
                projected_investor_graph,
                communities,
                weight="weight",
            )
        except Exception:
            summary["louvain_modularity"] = 0.0
        try:
            summary["avg_clustering"] = nx.average_clustering(projected_investor_graph, weight="weight")
        except Exception:
            summary["avg_clustering"] = 0.0
    else:
        summary["investor_communities"] = 0
        summary["louvain_modularity"] = 0.0
        summary["avg_clustering"] = 0.0

    return summary
