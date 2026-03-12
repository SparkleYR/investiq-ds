"""End-to-end pipeline for InvestIQ DataScience.

Run from the repository root with:
    python -m src.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .data_layer import build_deeptech_modules
from .graph_layer import build_bipartite_graph, compute_investor_metrics, compute_startup_metrics, project_investor_graph, summarize_graph
from .ml_layer import build_feature_matrix, build_temporal_split, train_random_forest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    deeptech_df, startups, investors, transactions = build_deeptech_modules()
    print(f"Deeptech rows: {len(deeptech_df):,}")
    print(f"Startups: {len(startups):,}")
    print(f"Investors: {len(investors):,}")
    print(f"Transactions: {len(transactions):,}")

    bipartite_graph = build_bipartite_graph(startups, investors, transactions)
    investor_graph = project_investor_graph(bipartite_graph)

    graph_summary = summarize_graph(bipartite_graph, investor_graph)
    print("\nGraph summary")
    for key, value in graph_summary.items():
        print(f"- {key}: {value}")

    investor_metrics = compute_investor_metrics(investor_graph)
    startup_metrics = compute_startup_metrics(bipartite_graph)

    investor_metrics.to_csv(output_dir / "investor_metrics.csv", index=False)
    startup_metrics.to_csv(output_dir / "startup_metrics.csv", index=False)

    split = build_temporal_split(
        startups=startups,
        investors=investors,
        transactions=transactions,
        negative_ratio=3.0,
    )
    feature_df = build_feature_matrix(split)
    feature_df.to_csv(output_dir / "link_prediction_features.csv", index=False)

    model, metrics, predictions = train_random_forest(feature_df)
    predictions.to_csv(output_dir / "link_prediction_predictions.csv", index=False)
    joblib.dump(model, output_dir / "rf_model.pkl")

    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (output_dir / "split_metadata.json").write_text(
        json.dumps(
            {
                "cutoff_date": str(split.cutoff_date),
                "num_feature_rows": int(len(feature_df)),
                "num_positive_labels": int(feature_df["label"].sum()),
                "num_negative_labels": int((feature_df["label"] == 0).sum()),
            },
            indent=2,
        )
    )

    feature_importance = pd.DataFrame(
        {
            "feature": [column for column in feature_df.columns if column not in {"label", "investor_id", "startup_id"}],
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    print("\nModel metrics")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}" if pd.notna(value) else f"- {key}: nan")

    print("\nTop predicted future links")
    print(predictions.head(10).to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
