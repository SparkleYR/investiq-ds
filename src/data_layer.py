"""
Data Layer — Entity Modules for InvestIQ DataScience

Provides three strict entity modules:
  • StartupModule  : startup_id, StartupName, SubVertical, City
  • InvestorModule : investor_id, InvestorsName, InvestorType
  • TransactionModule : edge_id, startup_id, investor_id, Date, AmountInUSD, InvestmentType

All normalization, deduplication, and cleaning happen here so downstream
graph / ML layers receive clean, ID-indexed DataFrames.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).resolve().parents[1] / ".." / "input" / "startup_funding.csv"

# Raw → canonical column rename
_COL_RENAME = {
    "Sr No": "SNo",
    "Date dd/mm/yyyy": "Date",
    "Startup Name": "StartupName",
    "Industry Vertical": "IndustryVertical",
    "SubVertical": "SubVertical",
    "City  Location": "City",
    "Investors Name": "InvestorsName",
    "InvestmentnType": "InvestmentType",
    "Amount in USD": "AmountInUSD",
    "Remarks": "Remarks",
}

# Funding-type normalization map
FUNDING_MAP: dict[str, str] = {
    "Seed/ Angel Funding": "Seed Angel Funding",
    "Seed / Angel Funding": "Seed Angel Funding",
    "Seed/Angel Funding": "Seed Angel Funding",
    "Angel / Seed Funding": "Seed Angel Funding",
    "Angel  Seed Funding": "Seed Angel Funding",
    "Seed  Angel Funding": "Seed Angel Funding",
    "Seed  Angle Funding": "Seed Angel Funding",
    "Seed Angle Funding": "Seed Angel Funding",
    "SeedAngel Funding": "Seed Angel Funding",
    "Seed\\nFunding": "Seed Funding",
    "SeednFunding": "Seed Funding",
    "Seed funding": "Seed Funding",
    "SeedFunding": "Seed Funding",
    "Seed Round": "Seed Funding",
    "Seed": "Seed Funding",
    "PrivateEquity": "Private Equity",
    "Private Equity Round": "Private Equity",
    "Crowd funding": "Crowd Funding",
    "preSeries A": "Pre-Series A",
    "preseries A": "Pre-Series A",
    "Pre Series A": "Pre-Series A",
    "PreSeries A": "Pre-Series A",
}

# Location normalization map (NCR cluster + Bengaluru alias)
LOCATION_MAP: dict[str, str] = {
    "Bengaluru": "Bangalore",
    "Delhi": "NCR",
    "New Delhi": "NCR",
    "Gurugram": "NCR",
    "Gurgaon": "NCR",
    "Noida": "NCR",
}

# Industry vertical normalization
INDUSTRY_MAP: dict[str, str] = {
    "eCommerce": "E-Commerce",
    "ECommerce": "E-Commerce",
    "Ecommerce": "E-Commerce",
    "E-commerce": "E-Commerce",
    "Ed-Tech": "EdTech",
    "Fin-Tech": "FinTech",
    "Food & Beverage": "Food and Beverage",
}

DEEPTECH_KEYWORDS: tuple[str, ...] = (
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "robotics",
    "space",
    "spacetech",
    "satellite",
    "aerospace",
    "defense",
    "drone",
    "autonomous",
    "semiconductor",
    "chip",
    "biotech",
    "bio-tech",
    "medtech",
    "healthtech",
    "cleantech",
    "climate tech",
    "iot",
    "industrial tech",
    "advanced materials",
)

# Investor type inference keywords
_INVESTOR_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"\b(venture|capital|partners|fund|vc)\b", "VC"),
    (r"\b(angel|individual)\b", "Angel"),
    (r"\b(private equity|pe fund)\b", "PE"),
    (r"\b(corporate|google|microsoft|intel|samsung|tata)\b", "Corporate"),
]

# Date edge-case fixes (raw string → corrected string)
_DATE_FIXES: dict[str, str] = {
    "01/07/015": "01/07/2015",
    "\\xc2\\xa010/7/2015": "10/07/2015",
    "12/05.2015": "12/05/2015",
    "13/04.2015": "13/04/2015",
    "15/01.2015": "15/01/2015",
    "22/01//2015": "22/01/2015",
    "05/072018": "05/07/2018",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_string(x: object) -> str:
    """Remove non-breaking-space artifacts."""
    return str(x).replace("\xc2\xa0", "").replace("\\xc2\\xa0", "").strip()


def _clean_amount(x: object) -> float:
    """Parse AmountInUSD to float; undisclosed/missing → -999 sentinel."""
    s = str(x).replace(",", "").replace("+", "").strip()
    s = s.lower().replace("undisclosed", "").replace("n/a", "")
    if s == "" or s == "nan":
        return -999.0
    try:
        return float(s)
    except ValueError:
        return -999.0


def _remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def _infer_investor_type(name: str) -> str:
    """Heuristic investor-type classification from name string."""
    lower = name.lower()
    for pattern, itype in _INVESTOR_TYPE_PATTERNS:
        if re.search(pattern, lower):
            return itype
    # Default to Unknown — can be manually enriched later
    return "Unknown"


def _is_deeptech_row(row: pd.Series, keywords: tuple[str, ...] = DEEPTECH_KEYWORDS) -> bool:
    """Heuristic deeptech flag based on IndustryVertical/SubVertical text."""
    industry = str(row.get("IndustryVertical", "")).lower()
    subvertical = str(row.get("SubVertical", "")).lower()
    combined = f"{industry} {subvertical}"
    return any(keyword in combined for keyword in keywords)


# ── Core loader ──────────────────────────────────────────────────────────────

def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load CSV and apply column rename + basic string cleaning.

    Returns a DataFrame with canonical column names and cleaned strings,
    but *no* entity splitting yet.
    """
    path = Path(path) if path else DATA_PATH
    df = pd.read_csv(path)
    df = df.rename(columns=_COL_RENAME)

    # String cleaning on all text columns
    text_cols = [
        "StartupName", "IndustryVertical", "SubVertical",
        "City", "InvestorsName", "InvestmentType", "AmountInUSD", "Remarks",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(_clean_string)

    # Date fixes
    for bad, good in _DATE_FIXES.items():
        df.loc[df["Date"] == bad, "Date"] = good

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

    # Amount cleaning
    df["CleanedAmount"] = df["AmountInUSD"].apply(_clean_amount)

    # Fix known outlier: Rapido INR→USD
    df.loc[df["CleanedAmount"] == 3_900_000_000, "CleanedAmount"] = 50_000_000

    # Normalize categoricals via loop (project convention: not .replace())
    for old, new in FUNDING_MAP.items():
        df.loc[df["InvestmentType"] == old, "InvestmentType"] = new

    # Remove residual punctuation from InvestmentType
    df["InvestmentType"] = df["InvestmentType"].apply(
        lambda x: _remove_punctuation(str(x)) if pd.notna(x) else x
    )
    # Re-apply funding map after punctuation removal
    for old, new in FUNDING_MAP.items():
        cleaned_old = _remove_punctuation(old)
        df.loc[df["InvestmentType"] == cleaned_old, "InvestmentType"] = new

    for old, new in LOCATION_MAP.items():
        df.loc[df["City"] == old, "City"] = new

    for old, new in INDUSTRY_MAP.items():
        df.loc[df["IndustryVertical"] == old, "IndustryVertical"] = new

    # Derived time columns
    df["Year"] = df["Date"].dt.year
    df["YearMonth"] = df["Date"].dt.to_period("M")

    return df


# ── Entity Module builders ───────────────────────────────────────────────────

def build_startup_module(df: pd.DataFrame) -> pd.DataFrame:
    """Startup Module: unique startups with metadata.

    Returns DataFrame indexed by `startup_id` with columns:
        StartupName, IndustryVertical, SubVertical, City
    """
    # Take the first occurrence of each startup for metadata
    startups = (
        df.groupby("StartupName")
        .agg({
            "IndustryVertical": "first",
            "SubVertical": "first",
            "City": "first",
        })
        .reset_index()
    )
    startups.index.name = "startup_id"
    startups = startups.reset_index()
    return startups


def build_investor_module(df: pd.DataFrame) -> pd.DataFrame:
    """Investor Module: unique investors with inferred type.

    Explodes comma-separated InvestorsName into individual rows.
    Returns DataFrame indexed by `investor_id` with columns:
        InvestorsName, InvestorType
    """
    # Explode multi-investor cells
    all_investors: list[str] = []
    for raw in df["InvestorsName"].dropna():
        for name in str(raw).split(","):
            cleaned = name.strip()
            if cleaned and cleaned.lower() not in ("", "nan", "undisclosed investors"):
                all_investors.append(cleaned)

    unique_names = sorted(set(all_investors))
    investors = pd.DataFrame({
        "InvestorsName": unique_names,
        "InvestorType": [_infer_investor_type(n) for n in unique_names],
    })
    investors.index.name = "investor_id"
    investors = investors.reset_index()
    return investors


def build_transaction_module(
    df: pd.DataFrame,
    startups: pd.DataFrame,
    investors: pd.DataFrame,
) -> pd.DataFrame:
    """Transaction Module: funding edges linking investors ↔ startups.

    Each row in the raw data may produce *multiple* edges (one per investor
    when InvestorsName is comma-separated).

    Returns DataFrame with columns:
        edge_id, startup_id, investor_id, Date, AmountInUSD, InvestmentType
    """
    # Build lookup dicts
    startup_lookup = {
        row["StartupName"]: row["startup_id"]
        for _, row in startups.iterrows()
    }
    investor_lookup = {
        row["InvestorsName"]: row["investor_id"]
        for _, row in investors.iterrows()
    }

    edges: list[dict] = []
    edge_id = 0
    for _, row in df.iterrows():
        s_id = startup_lookup.get(row["StartupName"])
        if s_id is None:
            continue

        inv_raw = str(row["InvestorsName"])
        for name in inv_raw.split(","):
            cleaned = name.strip()
            if cleaned.lower() in ("", "nan", "undisclosed investors"):
                continue
            i_id = investor_lookup.get(cleaned)
            if i_id is None:
                continue

            edges.append({
                "edge_id": edge_id,
                "startup_id": s_id,
                "investor_id": i_id,
                "Date": row["Date"],
                "AmountInUSD": row["CleanedAmount"],
                "InvestmentType": row["InvestmentType"],
            })
            edge_id += 1

    return pd.DataFrame(edges)


# ── Convenience: build all at once ───────────────────────────────────────────

def build_all_modules(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and return (raw_df, startups, investors, transactions)."""
    df = load_raw(path)
    startups = build_startup_module(df)
    investors = build_investor_module(df)
    transactions = build_transaction_module(df, startups, investors)
    return df, startups, investors, transactions


def filter_deeptech_transactions(
    df: pd.DataFrame,
    keywords: tuple[str, ...] = DEEPTECH_KEYWORDS,
) -> pd.DataFrame:
    """Return only rows that look like deeptech transactions.

    This is intentionally heuristic because the Kaggle dataset does not have a
    dedicated deeptech label.
    """
    mask = df.apply(lambda row: _is_deeptech_row(row, keywords=keywords), axis=1)
    return df.loc[mask].copy()


def build_deeptech_modules(
    path: str | Path | None = None,
    keywords: tuple[str, ...] = DEEPTECH_KEYWORDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data, filter for deeptech rows, and build all entity modules."""
    df = load_raw(path)
    deeptech_df = filter_deeptech_transactions(df, keywords=keywords)
    startups = build_startup_module(deeptech_df)
    investors = build_investor_module(deeptech_df)
    transactions = build_transaction_module(deeptech_df, startups, investors)
    return deeptech_df, startups, investors, transactions
