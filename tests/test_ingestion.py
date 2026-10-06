"""
test_ingestion.py — Unit tests for src/data/ingestion.py.

Tests cover:
  - load_csv: successful load, missing file error
  - validate_schema: valid data passes, invalid data raises SchemaError
  - ingest: end-to-end pipeline writes file to output directory
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

# Adjust import path so tests can be run from the project root with:
#   pytest tests/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.ingestion import load_csv, validate_schema, ingest, COFFEE_SALES_SCHEMA  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_CSV_CONTENT = textwrap.dedent("""\
    datetime,cash_type,card,money,coffee_name
    2024-01-01 08:00:00,card,CARD-001,3.50,Espresso
    2024-01-01 09:30:00,cash,,2.00,Americano
    2024-01-02 10:00:00,card,CARD-002,4.50,Latte
""")


@pytest.fixture()
def valid_csv_file(tmp_path: Path) -> Path:
    """Write a minimal valid CSV to a temp file and return its path."""
    csv_file = tmp_path / "coffee_sales.csv"
    csv_file.write_text(VALID_CSV_CONTENT, encoding="utf-8")
    return csv_file


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    """Return a validated DataFrame built from VALID_CSV_CONTENT."""
    return pd.read_csv(io.StringIO(VALID_CSV_CONTENT))


# ---------------------------------------------------------------------------
# load_csv tests
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_returns_dataframe(self, valid_csv_file: Path) -> None:
        df = load_csv(valid_csv_file)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, valid_csv_file: Path) -> None:
        df = load_csv(valid_csv_file)
        assert len(df) == 3

    def test_expected_columns_present(self, valid_csv_file: Path) -> None:
        df = load_csv(valid_csv_file)
        expected = {"datetime", "cash_type", "card", "money", "coffee_name"}
        assert expected.issubset(set(df.columns))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# validate_schema tests
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_data_passes(self, valid_df: pd.DataFrame) -> None:
        validated = validate_schema(valid_df)
        assert len(validated) == len(valid_df)

    def test_datetime_coercion(self, valid_df: pd.DataFrame) -> None:
        validated = validate_schema(valid_df)
        assert pd.api.types.is_datetime64_any_dtype(validated["datetime"])

    def test_money_coercion_to_float(self, valid_df: pd.DataFrame) -> None:
        validated = validate_schema(valid_df)
        assert pd.api.types.is_float_dtype(validated["money"])

    def test_null_card_allowed(self, valid_df: pd.DataFrame) -> None:
        """card column may have NaN values (anonymous customers)."""
        validated = validate_schema(valid_df)
        assert validated["card"].isna().any()

    def test_negative_money_fails(self, valid_df: pd.DataFrame) -> None:
        bad_df = valid_df.copy()
        bad_df.loc[0, "money"] = -1.0
        with pytest.raises(Exception):  # pandera.errors.SchemaError or SchemaErrors
            validate_schema(bad_df)

    def test_zero_money_fails(self, valid_df: pd.DataFrame) -> None:
        bad_df = valid_df.copy()
        bad_df.loc[0, "money"] = 0.0
        with pytest.raises(Exception):
            validate_schema(bad_df)

    def test_missing_required_column_fails(self, valid_df: pd.DataFrame) -> None:
        bad_df = valid_df.drop(columns=["coffee_name"])
        with pytest.raises(Exception):
            validate_schema(bad_df)

    def test_invalid_cash_type_fails(self, valid_df: pd.DataFrame) -> None:
        bad_df = valid_df.copy()
        bad_df.loc[0, "cash_type"] = "bitcoin"
        with pytest.raises(Exception):
            validate_schema(bad_df)

    def test_null_coffee_name_fails(self, valid_df: pd.DataFrame) -> None:
        bad_df = valid_df.copy()
        bad_df.loc[0, "coffee_name"] = None
        with pytest.raises(Exception):
            validate_schema(bad_df)


# ---------------------------------------------------------------------------
# ingest (end-to-end) tests
# ---------------------------------------------------------------------------


class TestIngest:
    def test_returns_dataframe(self, valid_csv_file: Path, tmp_path: Path) -> None:
        df = ingest(valid_csv_file, output_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_output_file_created(self, valid_csv_file: Path, tmp_path: Path) -> None:
        ingest(valid_csv_file, output_dir=tmp_path)
        assert (tmp_path / "coffee_sales.csv").exists()

    def test_output_row_count_matches_input(self, valid_csv_file: Path, tmp_path: Path) -> None:
        df = ingest(valid_csv_file, output_dir=tmp_path)
        saved = pd.read_csv(tmp_path / "coffee_sales.csv")
        assert len(df) == len(saved)

    def test_output_columns_preserved(self, valid_csv_file: Path, tmp_path: Path) -> None:
        df = ingest(valid_csv_file, output_dir=tmp_path)
        expected = {"datetime", "cash_type", "card", "money", "coffee_name"}
        assert expected.issubset(set(df.columns))

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ingest(tmp_path / "does_not_exist.csv", output_dir=tmp_path)
