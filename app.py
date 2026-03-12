from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_layer import build_deeptech_modules
from src.ml_layer import build_temporal_split, score_startup_investor_candidates

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "rf_model.pkl"
INVESTOR_METRICS_PATH = OUTPUT_DIR / "investor_metrics.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_project_data():
    deeptech_df, startups, investors, transactions = build_deeptech_modules()
    split = build_temporal_split(
        startups=startups,
        investors=investors,
        transactions=transactions,
        negative_ratio=3.0,
    )
    investor_metrics = pd.read_csv(INVESTOR_METRICS_PATH)
    return deeptech_df, startups, investors, transactions, split, investor_metrics


st.set_page_config(page_title="InvestIQ Deeptech Network", layout="wide")
st.title("InvestIQ — Indian Deeptech Investor Network")
st.caption("Phase 2 demo: ecosystem view and investor recommendation interface")

if not MODEL_PATH.exists():
    st.error("Model artifact not found. Run the pipeline first to generate outputs/rf_model.pkl.")
    st.stop()

if not INVESTOR_METRICS_PATH.exists():
    st.error("Investor metrics file not found. Run the pipeline first to generate outputs/investor_metrics.csv.")
    st.stop()

model = load_model()
deeptech_df, startups, investors, transactions, split, investor_metrics = load_project_data()

startup_names = sorted(startups["StartupName"].dropna().unique().tolist())
selected_startup_name = st.sidebar.selectbox("Select Startup", startup_names)
selected_startup_row = startups.loc[startups["StartupName"] == selected_startup_name].iloc[0]
selected_startup_id = int(selected_startup_row["startup_id"])

st.sidebar.markdown("### Selected Startup")
st.sidebar.write(f"**City:** {selected_startup_row.get('City', 'Unknown')}")
st.sidebar.write(f"**Industry:** {selected_startup_row.get('IndustryVertical', 'Unknown')}")
st.sidebar.write(f"**SubVertical:** {selected_startup_row.get('SubVertical', 'Unknown')}")

if METRICS_PATH.exists():
    model_metrics = pd.read_json(METRICS_PATH, typ="series")
    if "cv_roc_auc" in model_metrics:
        st.sidebar.metric("CV ROC-AUC", f"{float(model_metrics['cv_roc_auc']):.4f}")
    if "roc_auc" in model_metrics:
        st.sidebar.metric("Holdout ROC-AUC", f"{float(model_metrics['roc_auc']):.4f}")

tab1, tab2 = st.tabs(["Ecosystem View", "The Matchmaker"])

with tab1:
    st.subheader("Top 10 Investors by Degree Centrality")
    top_investors = investor_metrics.sort_values(by="degree_centrality", ascending=False).head(10)
    chart_df = top_investors[["name", "degree_centrality"]].set_index("name")
    st.bar_chart(chart_df)
    st.dataframe(
        top_investors[["name", "investor_type", "degree", "degree_centrality", "pagerank", "community_id"]],
        width="stretch",
        hide_index=True,
    )

with tab2:
    st.subheader(f"Recommended Investors for {selected_startup_name}")
    recommendations = score_startup_investor_candidates(
        model=model,
        split=split,
        transactions=transactions,
        startup_id=selected_startup_id,
        top_k=5,
    )

    if recommendations.empty:
        st.info("No recommendation candidates were available for this startup.")
    else:
        display_df = recommendations[
            [
                "InvestorsName",
                "InvestorType",
                "predicted_probability",
                "attribute_match_score",
                "industry_affinity",
                "city_affinity",
                "co_investor_weight_sum",
            ]
        ].rename(
            columns={
                "InvestorsName": "Recommended Investor",
                "InvestorType": "Investor Type",
                "predicted_probability": "Predicted Probability",
                "attribute_match_score": "Attribute Match Score",
                "industry_affinity": "Industry Affinity",
                "city_affinity": "City Affinity",
                "co_investor_weight_sum": "Co-Investment Signal",
            }
        )
        display_df["Predicted Probability"] = display_df["Predicted Probability"].round(4)
        display_df["Industry Affinity"] = display_df["Industry Affinity"].round(4)
        display_df["City Affinity"] = display_df["City Affinity"].round(4)
        display_df["Co-Investment Signal"] = display_df["Co-Investment Signal"].round(4)
        st.dataframe(display_df, width="stretch", hide_index=True)

        st.caption("Recommendations are generated from the saved Random Forest model and enriched graph + attribute features.")
