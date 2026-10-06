"""
rfm_features.py — Compute RFM (Recency, Frequency, Monetary) features for
each identified customer and persist the result as a Parquet feature store.

Input  : data/raw/coffee_sales.csv
Output : data/features/rfm_features.parquet

Columns produced
----------------
card            : customer identifier (anonymised loyalty-card token)
recency_days    : days since last purchase relative to dataset reference date
frequency       : total number of transactions in the full period
monetary        : total spend (sum of `money`) in the full period
recency_norm    : recency_days normalised to [0, 1] via Min-Max
frequency_norm  : frequency   normalised to [0, 1] via Min-Max
monetary_norm   : monetary    normalised to [0, 1] via Min-Max

Notes
-----
- Only rows with a non-null, non-"anonymous" card are included.
  ("anonymous" is the sentinel filled in by preprocessing.py for null cards.)
- Reference date = last observed transaction date in the dataset.
  Never uses datetime.today() so that results are fully reproducible.
- Recency is inverted for normalisation: higher recency_norm ↔ more recent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants (resolved relative to this file)
# __file__   = …/coffee_sales/src/features/rfm_features.py
# parents[0] = …/coffee_sales/src/features/
# parents[1] = …/coffee_sales/src/
# parents[2] = …/coffee_sales/          ← project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = _PROJECT_ROOT / "data" / "raw" / "coffee_sales.csv"
FEATURES_DIR = _PROJECT_ROOT / "data" / "features"
OUTPUT_PATH = FEATURES_DIR / "rfm_features.parquet"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM table from a raw transactions DataFrame.

    Parameters
    ----------
    df:
        Raw DataFrame containing at least the columns ``datetime``,
        ``card``, and ``money``.

    Returns
    -------
    pd.DataFrame
        One row per identified customer with columns:
        card, recency_days, frequency, monetary,
        recency_norm, frequency_norm, monetary_norm.
    """
    df = df.copy()

    # 1. Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # 2. Keep only identified customers (non-null, non-anonymous cards)
    df = df[df["card"].notna() & (df["card"] != "anonymous")].copy()
    logger.info("rfm: retained %d transactions for %d unique customers",
                len(df), df["card"].nunique())

    # 3. Reference date = last observed transaction date in the dataset
    reference_date = df["datetime"].max().normalize()  # floor to midnight
    logger.info("rfm: reference date = %s", reference_date.date())

    # 4. Aggregate per customer
    rfm = df.groupby("card", sort=False).agg(
        last_purchase=("datetime", "max"),
        frequency=("datetime", "count"),
        monetary=("money", "sum"),
    ).reset_index()

    rfm["recency_days"] = (reference_date - rfm["last_purchase"]).dt.days
    rfm = rfm.drop(columns=["last_purchase"])

    # 5. Min-Max normalisation
    #    Recency: lower recency_days → more recent → invert so higher norm = more recent
    rfm["recency_norm"] = _minmax_invert(rfm["recency_days"])
    rfm["frequency_norm"] = _minmax(rfm["frequency"])
    rfm["monetary_norm"] = _minmax(rfm["monetary"])

    # 6. Column order
    rfm = rfm[
        ["card", "recency_days", "frequency", "monetary",
         "recency_norm", "frequency_norm", "monetary_norm"]
    ]

    logger.info("rfm: final shape %s", rfm.shape)
    return rfm


def _minmax(series: pd.Series) -> pd.Series:
    """Min-Max scale a Series to [0, 1]. Returns 0.5 for constant series."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _minmax_invert(series: pd.Series) -> pd.Series:
    """Min-Max scale then invert (1 - scaled) so lower raw ↔ higher norm."""
    return 1.0 - _minmax(series)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read raw CSV → compute RFM → persist Parquet.

    Parameters
    ----------
    source_path : path to raw CSV (defaults to data/raw/coffee_sales.csv)
    output_path : destination Parquet path (defaults to data/features/rfm_features.parquet)

    Returns
    -------
    pd.DataFrame  The RFM feature table (also written to disk).
    """
    src = Path(source_path) if source_path else RAW_PATH
    dst = Path(output_path) if output_path else OUTPUT_PATH

    if not src.exists():
        raise FileNotFoundError(f"Raw dataset not found: {src}")

    logger.info("Loading raw data from %s", src)
    df_raw = pd.read_csv(src)

    rfm = compute_rfm(df_raw)

    dst.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(dst, index=False, engine="pyarrow")
    logger.info("RFM features written to %s (%d customers)", dst, len(rfm))

    return rfm


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    run()
