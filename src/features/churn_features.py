"""
Churn feature engineering.

Builds per-customer feature set for the churn classification model by combining:
  - RFM normalised scores (rfm_features.parquet)
  - Cluster / segment information (customer_segments.parquet)
  - Recency-based behavioural features derived from raw transactions
  - Binary churn label: client with no purchase in the last CHURN_WINDOW_DAYS
    before the reference date is labelled churn=1.

Usage:
    python -m src.features.churn_features
"""

import pandas as pd
from pathlib import Path

CHURN_WINDOW_DAYS: int = 90
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_CSV = DATA_DIR / "raw" / "coffee_sales.csv"
RFM_PARQUET = DATA_DIR / "features" / "rfm_features.parquet"
SEGMENTS_PARQUET = DATA_DIR / "features" / "customer_segments.parquet"
OUTPUT_PARQUET = DATA_DIR / "features" / "churn_features.parquet"


def build_churn_features(
    raw_csv: Path = RAW_CSV,
    rfm_parquet: Path = RFM_PARQUET,
    segments_parquet: Path = SEGMENTS_PARQUET,
    churn_window_days: int = CHURN_WINDOW_DAYS,
) -> pd.DataFrame:
    """Build the churn feature table for all identified customers.

    Parameters
    ----------
    raw_csv : Path
        Path to the raw transactions CSV.
    rfm_parquet : Path
        Path to rfm_features.parquet produced by Sub-Task 5.
    segments_parquet : Path
        Path to customer_segments.parquet produced by Sub-Task 5.
    churn_window_days : int
        Inactivity window in days that defines a churned customer.

    Returns
    -------
    pd.DataFrame
        One row per identified customer with columns:
        card, recency_norm, frequency_norm, monetary_norm, recency_days,
        frequency, monetary, cluster_id, segment_name,
        freq_last_4w, freq_last_8w, avg_ticket_recent,
        n_distinct_products, days_since_first_purchase, churn
    """
    # ------------------------------------------------------------------ #
    # 1. Load raw transactions — identified customers only                #
    # ------------------------------------------------------------------ #
    df = pd.read_csv(raw_csv)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Exclude rows without a card identifier (anonymous / cash-only)
    df = df[df["card"].notna()].copy()

    # ------------------------------------------------------------------ #
    # 2. Reference date = last recorded transaction (dataset end)         #
    # ------------------------------------------------------------------ #
    ref_date: pd.Timestamp = df["datetime"].max().normalize()

    # ------------------------------------------------------------------ #
    # 3. Churn label                                                       #
    # ------------------------------------------------------------------ #
    churn_cutoff = ref_date - pd.Timedelta(days=churn_window_days)
    last_purchase = df.groupby("card")["datetime"].max().rename("last_purchase_dt")
    churn_labels = (last_purchase < churn_cutoff).astype(int).rename("churn")

    # ------------------------------------------------------------------ #
    # 4. Recency-based frequency features                                 #
    # ------------------------------------------------------------------ #
    cutoff_4w = ref_date - pd.Timedelta(weeks=4)
    cutoff_8w = ref_date - pd.Timedelta(weeks=8)

    freq_4w = (
        df[df["datetime"] >= cutoff_4w]
        .groupby("card")
        .size()
        .rename("freq_last_4w")
    )
    freq_8w = (
        df[df["datetime"] >= cutoff_8w]
        .groupby("card")
        .size()
        .rename("freq_last_8w")
    )

    recent_txns = df[df["datetime"] >= cutoff_8w]
    avg_ticket_recent = (
        recent_txns.groupby("card")["money"].mean().rename("avg_ticket_recent")
    )

    # ------------------------------------------------------------------ #
    # 5. Historical diversity & tenure                                     #
    # ------------------------------------------------------------------ #
    n_distinct_products = (
        df.groupby("card")["coffee_name"].nunique().rename("n_distinct_products")
    )
    first_purchase = df.groupby("card")["datetime"].min().rename("first_purchase_dt")
    days_since_first = (
        (ref_date - first_purchase).dt.days.rename("days_since_first_purchase")
    )

    # ------------------------------------------------------------------ #
    # 6. Join RFM scores and segments                                      #
    # ------------------------------------------------------------------ #
    rfm = pd.read_parquet(rfm_parquet).set_index("card")
    segments = pd.read_parquet(segments_parquet)[
        ["card", "cluster_id", "segment_name"]
    ].set_index("card")

    features = (
        rfm[["recency_days", "frequency", "monetary",
             "recency_norm", "frequency_norm", "monetary_norm"]]
        .join(segments, how="inner")
        .join(churn_labels, how="inner")
        .join(freq_4w, how="left")
        .join(freq_8w, how="left")
        .join(avg_ticket_recent, how="left")
        .join(n_distinct_products, how="left")
        .join(days_since_first, how="left")
    )

    # Customers with no purchase in the recent windows → fill 0
    features["freq_last_4w"] = features["freq_last_4w"].fillna(0).astype(int)
    features["freq_last_8w"] = features["freq_last_8w"].fillna(0).astype(int)
    features["avg_ticket_recent"] = features["avg_ticket_recent"].fillna(0.0)

    features = features.reset_index()
    return features


if __name__ == "__main__":
    features = build_churn_features()
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PARQUET, index=False)
    churn_rate = features["churn"].mean()
    print(f"Saved {len(features):,} customers to {OUTPUT_PARQUET}")
    print(f"Churn rate: {churn_rate:.1%}  "
          f"({features['churn'].sum()} churned / {(1-churn_rate)*len(features):.0f} active)")
    print(features.head())
