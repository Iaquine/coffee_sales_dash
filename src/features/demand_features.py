"""
demand_features.py — Build daily time-series feature store for demand forecasting.

Input  : data/processed/transactions.parquet  (fallback: data/raw/coffee_sales.csv)
Output : data/features/demand_features.parquet

Feature schema per (product, date) row:
  - revenue          : sum of `money` for that day
  - n_transactions   : count of transactions
  - lag_1            : revenue 1 day prior
  - lag_7            : revenue 7 days prior
  - lag_14           : revenue 14 days prior
  - rolling_mean_7   : 7-day rolling average of revenue (window ends the day before)
  - rolling_mean_14  : 14-day rolling average of revenue
  - day_of_week      : 0=Monday … 6=Sunday
  - is_weekend       : 1 if Saturday or Sunday, else 0
  - month            : 1–12
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# __file__ is  …/coffee_sales/src/features/demand_features.py
# parents[0]  = …/coffee_sales/src/features/
# parents[1]  = …/coffee_sales/src/
# parents[2]  = …/coffee_sales/          ← project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_CSV = _PROJECT_ROOT / "data" / "raw" / "coffee_sales.csv"
_PROCESSED_PARQUET = _PROJECT_ROOT / "data" / "processed" / "transactions.parquet"
_FEATURES_DIR = _PROJECT_ROOT / "data" / "features"
_OUTPUT_PATH = _FEATURES_DIR / "demand_features.parquet"

FORECAST_PRODUCTS = [
    "Americano",
    "Americano with Milk",
    "Cappuccino",
    "Cocoa",
    "Cortado",
    "Espresso",
    "Hot Chocolate",
    "Latte",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _load_raw() -> pd.DataFrame:
    """Load transactions from processed Parquet or fall back to raw CSV."""
    if _PROCESSED_PARQUET.exists():
        logger.info("Loading from processed parquet: %s", _PROCESSED_PARQUET)
        df = pd.read_parquet(_PROCESSED_PARQUET, engine="pyarrow")
        # Ensure datetime column exists
        if "datetime" not in df.columns and "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])
        else:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    logger.warning("Processed parquet not found — falling back to raw CSV: %s", _RAW_CSV)
    df = pd.read_csv(_RAW_CSV, parse_dates=["datetime"])
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data into daily revenue + count per product.

    Parameters
    ----------
    df:
        Transaction-level DataFrame with columns ``datetime``, ``coffee_name``,
        ``money``.

    Returns
    -------
    pd.DataFrame
        Multi-indexed (product, date) with ``revenue`` and ``n_transactions``.
    """
    df = df.copy()
    df["date"] = df["datetime"].dt.normalize()  # midnight timestamp

    daily = (
        df.groupby(["coffee_name", "date"])
        .agg(revenue=("money", "sum"), n_transactions=("money", "count"))
        .reset_index()
        .rename(columns={"coffee_name": "product"})
    )
    # Keep only eligible forecast products
    daily = daily[daily["product"].isin(FORECAST_PRODUCTS)].copy()

    # Reindex to fill gaps with 0 so lag/rolling calculations are contiguous
    products = daily["product"].unique()
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    full_idx = pd.MultiIndex.from_product([products, all_dates], names=["product", "date"])
    daily = (
        daily.set_index(["product", "date"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )
    return daily


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag-1, lag-7, lag-14 revenue features per product."""
    df = df.sort_values(["product", "date"]).copy()
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df.groupby("product")["revenue"].shift(lag).fillna(0)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7-day and 14-day rolling mean of revenue (exclusive of current day)."""
    df = df.sort_values(["product", "date"]).copy()
    for window in [7, 14]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("product")["revenue"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .fillna(0)
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week, is_weekend, month calendar indicators."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    return df


def build_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full pipeline: aggregate → lag → rolling → calendar.

    Parameters
    ----------
    df:
        Optional pre-loaded raw/processed DataFrame. If None, data is loaded
        automatically from disk.

    Returns
    -------
    pd.DataFrame
        Feature store with one row per (product, date).
    """
    if df is None:
        df = _load_raw()

    daily = aggregate_daily(df)
    daily = add_lag_features(daily)
    daily = add_rolling_features(daily)
    daily = add_calendar_features(daily)

    logger.info(
        "demand_features built: %d rows, %d products, date range %s → %s",
        len(daily),
        daily["product"].nunique(),
        daily["date"].min().date(),
        daily["date"].max().date(),
    )
    return daily


def run(output_path: str | Path | None = None) -> pd.DataFrame:
    """Build features and persist to Parquet."""
    dst = Path(output_path) if output_path else _OUTPUT_PATH
    dst.parent.mkdir(parents=True, exist_ok=True)

    features = build_features()
    features.to_parquet(dst, index=False, engine="pyarrow")
    logger.info("demand_features saved to %s", dst)
    return features


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    run()
