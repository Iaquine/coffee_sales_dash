"""
ingestion.py — Load Kaggle Coffee Sales CSVs, combine, validate schema via
pandera, and persist a clean merged copy to data/raw/coffee_sales.csv.

The Kaggle dataset ships as two separate files:
  - index_1.csv  columns: date, datetime, cash_type, card, money, coffee_name
  - index_2.csv  columns: date, datetime, cash_type,       money, coffee_name
                 (cash-only transactions — no card column)

Both are merged into a single DataFrame where missing `card` values become NaN
(anonymous customers).  The combined file is then validated and persisted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition (applied after merging both source files)
# ---------------------------------------------------------------------------

COFFEE_SALES_SCHEMA = DataFrameSchema(
    columns={
        "datetime": Column(
            pa.DateTime,
            nullable=False,
            coerce=True,
            description="Timestamp of the transaction",
        ),
        "cash_type": Column(
            pa.String,
            nullable=False,
            checks=Check.isin(["card", "cash"]),
            description="Payment method: 'card' or 'cash'",
        ),
        "card": Column(
            pa.String,
            nullable=True,  # anonymous / cash customers have no card identifier
            description="Customer card identifier (null = anonymous)",
        ),
        "money": Column(
            pa.Float,
            nullable=False,
            coerce=True,
            checks=Check.greater_than(0),
            description="Transaction amount in local currency",
        ),
        "coffee_name": Column(
            pa.String,
            nullable=False,
            description="Name of the coffee product sold",
        ),
    },
    strict=False,  # allow extra columns (e.g. 'date') to pass through
    coerce=True,
    name="CoffeeSalesSchema",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

_REQUIRED_COLS = {"datetime", "cash_type", "money", "coffee_name"}


def _load_single_csv(path: Path) -> pd.DataFrame:
    """Read one CSV and normalise it to the canonical column set."""
    df = pd.read_csv(path)
    logger.info("  Read %s — %d rows, columns: %s", path.name, len(df), list(df.columns))
    # Ensure 'card' column exists even if absent (index_2.csv cash-only file)
    if "card" not in df.columns:
        df["card"] = pd.NA
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_csv(source_path: str | Path) -> pd.DataFrame:
    """Read a single coffee-sales CSV file and return a raw DataFrame.

    If *source_path* is the merged ``coffee_sales.csv`` it is read directly.
    Use :func:`load_and_merge` to combine the two Kaggle source files.

    Parameters
    ----------
    source_path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw data with the original column types preserved.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    logger.info("Reading CSV from %s", source_path)
    df = pd.read_csv(source_path)
    if "card" not in df.columns:
        df["card"] = pd.NA
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def load_and_merge(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load and merge the two Kaggle source files into one DataFrame.

    Looks for ``index_1.csv`` and ``index_2.csv`` inside *raw_dir*
    (defaults to ``data/raw/``).  At least one file must exist; if only one
    is found it is returned as-is after column normalisation.

    Parameters
    ----------
    raw_dir:
        Directory that contains ``index_1.csv`` and/or ``index_2.csv``.

    Returns
    -------
    pd.DataFrame
        Combined raw DataFrame with columns:
        datetime, cash_type, card (nullable), money, coffee_name
        plus any extra columns present in the source files (e.g. ``date``).
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    parts: list[pd.DataFrame] = []

    for fname in ("index_1.csv", "index_2.csv"):
        fpath = raw_dir / fname
        if fpath.exists():
            parts.append(_load_single_csv(fpath))
        else:
            logger.warning("Source file not found, skipping: %s", fpath)

    if not parts:
        raise FileNotFoundError(
            f"No source CSV files found in {raw_dir}. "
            "Expected index_1.csv and/or index_2.csv."
        )

    df = pd.concat(parts, ignore_index=True)
    logger.info("Merged %d source file(s) → %d total rows", len(parts), len(df))
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce DataFrame against CoffeeSalesSchema.

    Parameters
    ----------
    df:
        Raw DataFrame from :func:`load_csv` or :func:`load_and_merge`.

    Returns
    -------
    pd.DataFrame
        Validated (and coerced) DataFrame.

    Raises
    ------
    pandera.errors.SchemaError
        If the DataFrame does not satisfy the schema constraints.
    """
    # Pre-parse datetime with mixed formats before pandera coercion to avoid
    # issues when index_1 (microsecond timestamps) and index_2 (second-level
    # timestamps) are merged into the same column.
    if "datetime" in df.columns and df["datetime"].dtype == object:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", utc=False)

    logger.info("Validating schema …")
    validated = COFFEE_SALES_SCHEMA.validate(df, lazy=True)
    logger.info("Schema validation passed — %d rows validated", len(validated))
    return validated


def ingest(
    source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Full ingestion pipeline: load (+ merge) → validate → persist.

    Behaviour
    ---------
    * If *source_path* is given, reads that single file.
    * Otherwise calls :func:`load_and_merge` to combine ``index_1.csv`` +
      ``index_2.csv`` from *raw_dir* (or the default ``data/raw/``).

    The validated DataFrame is written to ``<output_dir>/coffee_sales.csv``.

    Parameters
    ----------
    source_path:
        Path to a single merged CSV. If ``None``, auto-merges source files.
    output_dir:
        Destination directory for the merged CSV.
        Defaults to ``data/raw/`` relative to the project root.
    raw_dir:
        Source directory for :func:`load_and_merge`.
        Ignored when *source_path* is provided.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame (post-coercion).
    """
    if source_path is not None:
        df_raw = load_csv(source_path)
    else:
        df_raw = load_and_merge(raw_dir)

    df_validated = validate_schema(df_raw)

    out_dir = Path(output_dir) if output_dir else RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coffee_sales.csv"

    df_validated.to_csv(out_path, index=False)
    logger.info("Validated merged dataset persisted to %s (%d rows)", out_path, len(df_validated))

    return df_validated
