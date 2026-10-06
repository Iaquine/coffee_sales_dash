"""
preprocessing.py — Clean and enrich raw coffee-sales data, producing the
processed layer used by all downstream models.

Input  : data/raw/coffee_sales.csv  (validated schema from ingestion.py)
Output : data/processed/transactions.parquet

Pipeline steps (in order):
  1. Parse `datetime` column to pandas Timestamp (UTC-naive)
  2. Extract temporal features:
       hour, day_of_week (0=Mon), week_of_year, month, year,
       is_weekend (bool), time_of_day_bucket (morning/afternoon/evening/night)
  3. Remove rows where money <= 0 or coffee_name is null
  4. Identify and remove monetary outliers via IQR (1.5× fence), logging count
  5. Normalise `card`: fill nulls with the string "anonymous"
  6. Persist result to data/processed/transactions.parquet via PyArrow engine
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants (resolved relative to this file so imports work from any cwd)
# ---------------------------------------------------------------------------

# __file__   = …/coffee_sales/src/data/preprocessing.py
# parents[0] = …/coffee_sales/src/data/
# parents[1] = …/coffee_sales/src/
# parents[2] = …/coffee_sales/          ← project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = _PROJECT_ROOT / "data" / "raw" / "coffee_sales.csv"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "transactions.parquet"

# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1 — Parse the 'datetime' column to pandas Timestamp.

    Coerces strings with :func:`pandas.to_datetime`. Rows where the value
    cannot be parsed become NaT and are dropped with a warning.
    """
    df = df.copy()
    original_len = len(df)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    nat_count = df["datetime"].isna().sum()
    if nat_count:
        logger.warning("Dropping %d rows with unparseable datetime values", nat_count)
        df = df.dropna(subset=["datetime"])
    logger.info("parse_datetime: %d → %d rows", original_len, len(df))
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 — Derive temporal features from the parsed 'datetime' column.

    New columns added:
    - ``hour``             : 0–23
    - ``day_of_week``      : 0 (Monday) – 6 (Sunday)
    - ``week_of_year``     : ISO week number (1–53)
    - ``month``            : 1–12
    - ``year``             : e.g. 2024
    - ``is_weekend``       : True for Saturday (5) and Sunday (6)
    - ``time_of_day_bucket``: one of morning / afternoon / evening / night
    """
    df = df.copy()
    dt = df["datetime"].dt

    df["hour"] = dt.hour.astype("int8")
    df["day_of_week"] = dt.dayofweek.astype("int8")          # 0=Mon … 6=Sun
    df["week_of_year"] = dt.isocalendar().week.astype("int8")
    df["month"] = dt.month.astype("int8")
    df["year"] = dt.year.astype("int16")
    df["is_weekend"] = df["day_of_week"] >= 5                # Sat=5, Sun=6

    df["time_of_day_bucket"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 20, 23],
        labels=["night", "morning", "afternoon", "evening", "night"],
        ordered=False,
    )
    # pd.cut with duplicate labels leaves the last bin's label; remap explicitly
    # for the wraparound hours 21-23 → night:
    df["time_of_day_bucket"] = df["hour"].map(_hour_to_bucket).astype("category")

    logger.info("extract_temporal_features: added 7 temporal columns")
    return df


def _hour_to_bucket(hour: int) -> str:
    """Map a single hour value to a time-of-day bucket label."""
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    if 18 <= hour <= 20:
        return "evening"
    return "night"  # 21-23 and 0-5


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 — Drop rows with money <= 0 or null coffee_name."""
    original_len = len(df)
    df = df[df["money"] > 0].copy()
    df = df.dropna(subset=["coffee_name"])
    removed = original_len - len(df)
    if removed:
        logger.info("remove_invalid_rows: dropped %d rows (money<=0 or null coffee_name)", removed)
    return df


def remove_monetary_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 — Remove monetary outliers using the IQR × 1.5 fence.

    Only the upper fence is applied (extreme high values); negative/zero values
    are already handled in step 3. The number of removed rows is logged.
    """
    q1 = df["money"].quantile(0.25)
    q3 = df["money"].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    lower_fence = q1 - 1.5 * iqr

    original_len = len(df)
    mask = (df["money"] >= lower_fence) & (df["money"] <= upper_fence)
    df = df[mask].copy()
    removed = original_len - len(df)
    logger.info(
        "remove_monetary_outliers: IQR fence [%.2f, %.2f] — removed %d outlier rows",
        lower_fence,
        upper_fence,
        removed,
    )
    return df


def normalise_card(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5 — Fill null values in the 'card' column with 'anonymous'."""
    df = df.copy()
    null_count = df["card"].isna().sum()
    df["card"] = df["card"].fillna("anonymous")
    logger.info("normalise_card: filled %d null card values with 'anonymous'", null_count)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a raw coffee-sales DataFrame.

    Parameters
    ----------
    df:
        Raw DataFrame that has already passed schema validation (see
        :mod:`src.data.ingestion`). The DataFrame is not mutated in place.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-enriched DataFrame ready for downstream modelling.
    """
    df = parse_datetime(df)
    df = extract_temporal_features(df)
    df = remove_invalid_rows(df)
    df = remove_monetary_outliers(df)
    df = normalise_card(df)
    logger.info("preprocess complete: %d rows in final processed dataset", len(df))
    return df


def run(
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """End-to-end entry point: read CSV → preprocess → persist Parquet.

    Parameters
    ----------
    source_path:
        Path to the raw CSV file. Defaults to ``data/raw/coffee_sales.csv``
        relative to the project root.
    output_path:
        Destination for the Parquet file. Defaults to
        ``data/processed/transactions.parquet``.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame (same object written to disk).
    """
    src = Path(source_path) if source_path else RAW_PATH
    dst = Path(output_path) if output_path else PROCESSED_PATH

    if not src.exists():
        raise FileNotFoundError(f"Raw dataset not found: {src}")

    logger.info("Loading raw data from %s", src)
    df_raw = pd.read_csv(src)

    df_processed = preprocess(df_raw)

    dst.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(dst, index=False, engine="pyarrow")
    logger.info("Processed dataset written to %s (%d rows)", dst, len(df_processed))

    return df_processed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    run()
