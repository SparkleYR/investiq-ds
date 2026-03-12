"""
Machine Learning Layer — Link Prediction Engine.

This module treats future funding as an investor↔startup link prediction task.
It builds temporal train/evaluation candidate pairs, extracts graph-based
features, and trains a Random Forest classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .graph_layer import (
    build_bipartite_graph,
    investor_node_id,
    project_investor_graph,
    startup_node_id,
)


@dataclass
class TemporalLinkPredictionSplit:
    cutoff_date: pd.Timestamp
    startups: pd.DataFrame
    investors: pd.DataFrame
    graph_before: nx.Graph
    investor_projection_before: nx.Graph
    historical_transactions: pd.DataFrame
    future_transactions: pd.DataFrame
    labeled_pairs: pd.DataFrame


@dataclass
class FeatureContext:
    startup_lookup: dict[int, dict[str, object]]
    investor_profiles: dict[int, dict[str, object]]


def _existing_pairs(transactions: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(transactions["investor_id"], transactions["startup_id"]))


def _mode_or_unknown(series: pd.Series) -> str:
    cleaned = series.dropna().astype(str)
    cleaned = cleaned[cleaned.str.lower() != "nan"]
    if cleaned.empty:
        return "Unknown"
    modes = cleaned.mode()
    return str(modes.iloc[0]) if not modes.empty else str(cleaned.iloc[0])


def _normalized_match(left: object, right: object) -> int:
    left_value = str(left).strip().lower()
    right_value = str(right).strip().lower()
    if left_value in {"", "nan", "unknown"} or right_value in {"", "nan", "unknown"}:
        return 0
    return int(left_value == right_value)


def build_feature_context(
    startups: pd.DataFrame,
    historical_transactions: pd.DataFrame,
) -> FeatureContext:
    """Precompute startup metadata and investor historical portfolio profiles."""
    startup_lookup = (
        startups[["startup_id", "StartupName", "City", "IndustryVertical", "SubVertical"]]
        .set_index("startup_id")
        .to_dict("index")
    )

    portfolio_df = historical_transactions.merge(
        startups[["startup_id", "City", "IndustryVertical", "SubVertical"]],
        on="startup_id",
        how="left",
    )

    investor_profiles: dict[int, dict[str, object]] = {}
    for investor_id, group in portfolio_df.groupby("investor_id"):
        city_counts = group["City"].fillna("Unknown").astype(str).value_counts().to_dict()
        industry_counts = group["IndustryVertical"].fillna("Unknown").astype(str).value_counts().to_dict()
        subvertical_counts = group["SubVertical"].fillna("Unknown").astype(str).value_counts().to_dict()
        portfolio_size = max(int(len(group)), 1)
        mean_amount = 0.0
        if "AmountInUSD" in group.columns:
            clean_amounts = group["AmountInUSD"].replace(-999, np.nan).dropna()
            mean_amount = float(clean_amounts.mean()) if not clean_amounts.empty else 0.0

        investor_profiles[int(investor_id)] = {
            "portfolio_size": portfolio_size,
            "city_mode": _mode_or_unknown(group["City"]),
            "industry_mode": _mode_or_unknown(group["IndustryVertical"]),
            "subvertical_mode": _mode_or_unknown(group["SubVertical"]),
            "city_counts": city_counts,
            "industry_counts": industry_counts,
            "subvertical_counts": subvertical_counts,
            "mean_amount": mean_amount,
        }

    return FeatureContext(startup_lookup=startup_lookup, investor_profiles=investor_profiles)


def _count_future_positive_pairs(transactions: pd.DataFrame, cutoff_date: pd.Timestamp) -> int:
    historical_transactions = transactions[transactions["Date"] <= cutoff_date].copy()
    future_transactions = transactions[transactions["Date"] > cutoff_date].copy()

    active_investors = set(historical_transactions["investor_id"].dropna().astype(int).unique())
    active_startups = set(historical_transactions["startup_id"].dropna().astype(int).unique())
    historical_pairs = _existing_pairs(historical_transactions)
    future_pairs = _existing_pairs(future_transactions)

    positive_pairs = [
        pair
        for pair in future_pairs
        if pair not in historical_pairs
        and pair[0] in active_investors
        and pair[1] in active_startups
    ]
    return len(positive_pairs)


def choose_cutoff_date(
    transactions: pd.DataFrame,
    quantile: float = 0.8,
    min_positive_pairs: int = 4,
) -> pd.Timestamp:
    """Choose a temporal cutoff with enough future positive links when possible."""
    dated = transactions.dropna(subset=["Date"]).sort_values("Date")
    if dated.empty:
        raise ValueError("No valid transaction dates found.")

    candidate_quantiles = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    viable_cutoffs: list[tuple[int, pd.Timestamp]] = []
    for candidate_quantile in candidate_quantiles:
        candidate_cutoff = dated["Date"].quantile(candidate_quantile)
        positive_count = _count_future_positive_pairs(transactions, candidate_cutoff)
        viable_cutoffs.append((positive_count, pd.Timestamp(candidate_cutoff)))
        if positive_count >= min_positive_pairs:
            return pd.Timestamp(candidate_cutoff)

    if viable_cutoffs:
        viable_cutoffs.sort(key=lambda item: item[0], reverse=True)
        return viable_cutoffs[0][1]

    return pd.Timestamp(dated["Date"].quantile(quantile))


def build_temporal_split(
    startups: pd.DataFrame,
    investors: pd.DataFrame,
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
    negative_ratio: float = 1.0,
    random_state: int = 42,
) -> TemporalLinkPredictionSplit:
    """Create a temporal split suitable for future-link prediction."""
    if cutoff_date is None:
        cutoff_date = choose_cutoff_date(transactions)

    historical_transactions = transactions[transactions["Date"] <= cutoff_date].copy()
    future_transactions = transactions[transactions["Date"] > cutoff_date].copy()

    active_investors = set(historical_transactions["investor_id"].dropna().astype(int).unique())
    active_startups = set(historical_transactions["startup_id"].dropna().astype(int).unique())
    historical_pairs = _existing_pairs(historical_transactions)
    future_pairs = _existing_pairs(future_transactions)

    positive_pairs = sorted(
        pair
        for pair in future_pairs
        if pair not in historical_pairs
        and pair[0] in active_investors
        and pair[1] in active_startups
    )
    if not positive_pairs:
        raise ValueError("No future positive links found after the selected cutoff date.")

    active_investors = sorted(active_investors)
    active_startups = sorted(active_startups)
    candidate_negatives = [
        (investor_id, startup_id)
        for investor_id, startup_id in product(active_investors, active_startups)
        if (investor_id, startup_id) not in historical_pairs
        and (investor_id, startup_id) not in future_pairs
    ]
    if not candidate_negatives:
        raise ValueError("No candidate negative pairs available for link prediction.")

    rng = np.random.default_rng(random_state)
    negative_sample_size = min(len(candidate_negatives), max(1, int(len(positive_pairs) * negative_ratio)))
    negative_indices = rng.choice(len(candidate_negatives), size=negative_sample_size, replace=False)
    negative_pairs = [candidate_negatives[index] for index in negative_indices]

    labeled_pairs = pd.DataFrame(
        [{"investor_id": i, "startup_id": s, "label": 1} for i, s in positive_pairs]
        + [{"investor_id": i, "startup_id": s, "label": 0} for i, s in negative_pairs]
    )

    graph_before = build_bipartite_graph(startups, investors, historical_transactions)
    investor_projection_before = project_investor_graph(graph_before)

    return TemporalLinkPredictionSplit(
        cutoff_date=pd.Timestamp(cutoff_date),
        startups=startups,
        investors=investors,
        graph_before=graph_before,
        investor_projection_before=investor_projection_before,
        historical_transactions=historical_transactions,
        future_transactions=future_transactions,
        labeled_pairs=labeled_pairs,
    )


def extract_pair_features(
    graph_before: nx.Graph,
    investor_projection_before: nx.Graph,
    feature_context: FeatureContext,
    investor_id: int,
    startup_id: int,
) -> dict[str, float | int]:
    """Extract enriched graph + attribute features for a candidate pair."""
    investor_node = investor_node_id(investor_id)
    startup_node = startup_node_id(startup_id)

    startup_attrs = feature_context.startup_lookup.get(startup_id, {})
    investor_profile = feature_context.investor_profiles.get(investor_id, {})
    portfolio_size = max(int(investor_profile.get("portfolio_size", 0)), 1)

    target_city = startup_attrs.get("City", "Unknown")
    target_industry = startup_attrs.get("IndustryVertical", "Unknown")
    target_subvertical = startup_attrs.get("SubVertical", "Unknown")

    city_mode_match = _normalized_match(target_city, investor_profile.get("city_mode", "Unknown"))
    industry_mode_match = _normalized_match(target_industry, investor_profile.get("industry_mode", "Unknown"))
    subvertical_mode_match = _normalized_match(target_subvertical, investor_profile.get("subvertical_mode", "Unknown"))
    city_affinity = investor_profile.get("city_counts", {}).get(str(target_city), 0) / portfolio_size
    industry_affinity = investor_profile.get("industry_counts", {}).get(str(target_industry), 0) / portfolio_size
    subvertical_affinity = investor_profile.get("subvertical_counts", {}).get(str(target_subvertical), 0) / portfolio_size
    attribute_match_score = city_mode_match + industry_mode_match

    if investor_node not in graph_before or startup_node not in graph_before:
        return {
            "investor_id": investor_id,
            "startup_id": startup_id,
            "investor_degree": 0,
            "startup_degree": 0,
            "preferential_attachment": 0,
            "common_neighbors": 0,
            "jaccard": 0.0,
            "adamic_adar": 0.0,
            "resource_allocation": 0.0,
            "bipartite_resource_allocation": 0.0,
            "bipartite_adamic_adar": 0.0,
            "co_investor_weight_sum": 0.0,
            "co_investor_weight_mean": 0.0,
            "co_investor_weight_max": 0.0,
            "connected_current_investors": 0,
            "startup_current_investors": 0,
            "investor_co_investor_degree": 0,
            "portfolio_overlap": 0,
            "portfolio_jaccard": 0.0,
            "city_mode_match": city_mode_match,
            "industry_mode_match": industry_mode_match,
            "subvertical_mode_match": subvertical_mode_match,
            "city_affinity": city_affinity,
            "industry_affinity": industry_affinity,
            "subvertical_affinity": subvertical_affinity,
            "attribute_match_score": attribute_match_score,
            "investor_portfolio_size": portfolio_size,
        }

    investor_degree = graph_before.degree(investor_node)
    startup_degree = graph_before.degree(startup_node)
    startup_current_investors = set(graph_before.neighbors(startup_node))
    candidate_startups = set(graph_before.neighbors(investor_node))

    investor_projection_neighbors = set(investor_projection_before.neighbors(investor_node)) if investor_node in investor_projection_before else set()
    intersection = investor_projection_neighbors.intersection(startup_current_investors)
    union = investor_projection_neighbors.union(startup_current_investors)
    common_neighbors = len(intersection)
    jaccard = common_neighbors / len(union) if union else 0.0

    adamic_adar = 0.0
    resource_allocation = 0.0
    for node in intersection:
        degree = investor_projection_before.degree(node)
        if degree > 1:
            adamic_adar += 1 / math.log(degree)
        if degree > 0:
            resource_allocation += 1 / degree

    co_investor_weights: list[float] = []
    bipartite_resource_allocation = 0.0
    bipartite_adamic_adar = 0.0
    syndicate_portfolio: set[str] = set()
    for current_investor in startup_current_investors:
        weight = 0.0
        if investor_projection_before.has_edge(investor_node, current_investor):
            weight = float(investor_projection_before[investor_node][current_investor].get("weight", 1.0))
        co_investor_weights.append(weight)

        weighted_degree = investor_projection_before.degree(current_investor, weight="weight")
        if weight > 0 and weighted_degree > 0:
            bipartite_resource_allocation += weight / weighted_degree
            bipartite_adamic_adar += weight / math.log1p(weighted_degree)

        syndicate_portfolio.update(graph_before.neighbors(current_investor))

    syndicate_portfolio.discard(startup_node)
    portfolio_overlap = len(candidate_startups.intersection(syndicate_portfolio))
    portfolio_union = candidate_startups.union(syndicate_portfolio)
    portfolio_jaccard = portfolio_overlap / len(portfolio_union) if portfolio_union else 0.0

    co_investor_weight_sum = float(sum(co_investor_weights))
    co_investor_weight_mean = co_investor_weight_sum / len(co_investor_weights) if co_investor_weights else 0.0
    co_investor_weight_max = float(max(co_investor_weights)) if co_investor_weights else 0.0
    connected_current_investors = int(sum(weight > 0 for weight in co_investor_weights))

    return {
        "investor_id": investor_id,
        "startup_id": startup_id,
        "investor_degree": investor_degree,
        "startup_degree": startup_degree,
        "preferential_attachment": investor_degree * startup_degree,
        "common_neighbors": common_neighbors,
        "jaccard": jaccard,
        "adamic_adar": adamic_adar,
        "resource_allocation": resource_allocation,
        "bipartite_resource_allocation": bipartite_resource_allocation,
        "bipartite_adamic_adar": bipartite_adamic_adar,
        "co_investor_weight_sum": co_investor_weight_sum,
        "co_investor_weight_mean": co_investor_weight_mean,
        "co_investor_weight_max": co_investor_weight_max,
        "connected_current_investors": connected_current_investors,
        "startup_current_investors": len(startup_current_investors),
        "investor_co_investor_degree": len(investor_projection_neighbors),
        "portfolio_overlap": portfolio_overlap,
        "portfolio_jaccard": portfolio_jaccard,
        "city_mode_match": city_mode_match,
        "industry_mode_match": industry_mode_match,
        "subvertical_mode_match": subvertical_mode_match,
        "city_affinity": city_affinity,
        "industry_affinity": industry_affinity,
        "subvertical_affinity": subvertical_affinity,
        "attribute_match_score": attribute_match_score,
        "investor_portfolio_size": portfolio_size,
    }


def build_feature_matrix(split: TemporalLinkPredictionSplit) -> pd.DataFrame:
    """Build a labeled feature matrix for model training/evaluation."""
    context = build_feature_context(split.startups, split.historical_transactions)
    rows = []
    for _, pair in split.labeled_pairs.iterrows():
        features = extract_pair_features(
            graph_before=split.graph_before,
            investor_projection_before=split.investor_projection_before,
            feature_context=context,
            investor_id=int(pair["investor_id"]),
            startup_id=int(pair["startup_id"]),
        )
        features["label"] = int(pair["label"])
        rows.append(features)

    return pd.DataFrame(rows).fillna(0)


def train_random_forest(
    feature_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict[str, float], pd.DataFrame]:
    """Train a tuned Random Forest model and return model, metrics, predictions."""
    if feature_df.empty:
        raise ValueError("Feature dataframe is empty.")

    model_features = [column for column in feature_df.columns if column not in {"label", "investor_id", "startup_id"}]
    X = feature_df[model_features].fillna(0)
    y = feature_df["label"]
    if y.nunique() < 2:
        raise ValueError("Link prediction dataset contains only one class; adjust cutoff or sampling.")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=random_state,
        class_weight="balanced",
    )

    metrics: dict[str, float] = {}
    if len(feature_df) >= 10 and y.value_counts().min() >= 2:
        cv_splits = min(3, int(y.value_counts().min()))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        cv_probabilities = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        positive_class_index = 1 if cv_probabilities.shape[1] > 1 else 0
        metrics["cv_roc_auc"] = roc_auc_score(y, cv_probabilities[:, positive_class_index])

    stratify = y if y.value_counts().min() > 1 else None
    X_train, X_test, y_train, y_test, _, meta_test = train_test_split(
        X,
        y,
        feature_df[["investor_id", "startup_id"]],
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob_all = model.predict_proba(X_test)
    positive_class_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
    y_prob = y_prob_all[:, positive_class_index]

    metrics.update(
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else float("nan"),
        }
    )

    predictions = meta_test.copy()
    predictions["label"] = y_test.values
    predictions["predicted_label"] = y_pred
    predictions["predicted_probability"] = y_prob
    predictions = predictions.sort_values(by="predicted_probability", ascending=False)

    model.fit(X, y)
    return model, metrics, predictions


def score_startup_investor_candidates(
    model: RandomForestClassifier,
    split: TemporalLinkPredictionSplit,
    transactions: pd.DataFrame,
    startup_id: int,
    top_k: int = 5,
) -> pd.DataFrame:
    """Score all investors not already connected to a startup anywhere in the dataset."""
    context = build_feature_context(split.startups, split.historical_transactions)
    existing_pairs = _existing_pairs(transactions)
    candidate_investor_ids = [
        int(investor_id)
        for investor_id in split.investors["investor_id"].tolist()
        if (int(investor_id), int(startup_id)) not in existing_pairs
    ]

    rows = [
        extract_pair_features(
            graph_before=split.graph_before,
            investor_projection_before=split.investor_projection_before,
            feature_context=context,
            investor_id=investor_id,
            startup_id=int(startup_id),
        )
        for investor_id in candidate_investor_ids
    ]
    candidate_df = pd.DataFrame(rows).fillna(0)
    if candidate_df.empty:
        return candidate_df

    model_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else [
        column for column in candidate_df.columns if column not in {"investor_id", "startup_id"}
    ]
    probabilities = model.predict_proba(candidate_df[model_features])
    positive_class_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
    candidate_df["predicted_probability"] = probabilities[:, positive_class_index]

    candidate_df = candidate_df.merge(
        split.investors[["investor_id", "InvestorsName", "InvestorType"]],
        on="investor_id",
        how="left",
    )
    candidate_df = candidate_df.merge(
        split.startups[["startup_id", "StartupName", "City", "IndustryVertical"]],
        on="startup_id",
        how="left",
    )
    return candidate_df.sort_values(by="predicted_probability", ascending=False).head(top_k)


def rank_candidate_links(
    graph_before: nx.Graph,
    investor_projection_before: nx.Graph,
    split: TemporalLinkPredictionSplit,
    investor_ids: list[int],
    startup_ids: list[int],
    top_k: int = 20,
) -> pd.DataFrame:
    """Score arbitrary investor↔startup candidate pairs using enriched heuristics only."""
    context = build_feature_context(split.startups, split.historical_transactions)
    records = [
        extract_pair_features(
            graph_before=graph_before,
            investor_projection_before=investor_projection_before,
            feature_context=context,
            investor_id=investor_id,
            startup_id=startup_id,
        )
        for investor_id, startup_id in product(investor_ids, startup_ids)
        if not graph_before.has_edge(investor_node_id(investor_id), startup_node_id(startup_id))
    ]
    scored = pd.DataFrame(records)
    if scored.empty:
        return scored

    scored["heuristic_score"] = (
        scored["attribute_match_score"]
        + scored["industry_affinity"]
        + scored["city_affinity"]
        + scored["bipartite_resource_allocation"]
        + scored["bipartite_adamic_adar"]
        + scored["co_investor_weight_sum"]
    )
    return scored.sort_values(by="heuristic_score", ascending=False).head(top_k)
