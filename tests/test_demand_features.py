"""
test_demand_features.py — Unit tests for src/features/demand_features.py

Coverage:
  - aggregate_daily: grouping by product and date, zero-fill for missing days
  - add_lag_features: correct shift per product, no future leakage
  - add_rolling_features: rolling mean excludes current day (shift(1))
  - add_calendar_features: day_of_week, is_weekend, month values
  - build_features: end-to-end shape and required columns
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.demand_features import (
    FORECAST_PRODUCTS,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    aggregate_daily,
    build_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_transactions(
    products: list[str] | None = None,
    n_days: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic transaction DataFrame (only columns required by demand features)."""
    rng = np.random.default_rng(seed)
    if products is None:
        products = ["Latte", "Espresso"]

    base_date = pd.Timestamp("2024-01-01")
    rows = []
    for day_offset in range(n_days):
        date = base_date + pd.Timedelta(days=day_offset)
        for product in products:
            n_txns = int(rng.integers(1, 6))
            for _ in range(n_txns):
                rows.append(
                    {
                        "datetime": date + pd.Timedelta(hours=int(rng.integers(8, 20))),
                        "coffee_name": product,
                        "money": float(rng.uniform(3.0, 8.0)),
                        "card": "CARD_001",
                        "cash_type": "card",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def txn_df() -> pd.DataFrame:
    return _make_transactions(products=["Latte", "Espresso"], n_days=30)


@pytest.fixture()
def daily_df(txn_df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_daily(txn_df)


# ---------------------------------------------------------------------------
# Tests — aggregate_daily
# ---------------------------------------------------------------------------


class TestAggregateDaily:
    def test_output_columns(self, daily_df):
        assert {"product", "date", "revenue", "n_transactions"}.issubset(daily_df.columns)

    def test_one_row_per_product_per_day(self, txn_df):
        daily = aggregate_daily(txn_df)
        dupes = daily.duplicated(subset=["product", "date"])
        assert not dupes.any(), "Found duplicate (product, date) rows"

    def test_revenue_is_positive_for_active_days(self, daily_df):
        # Days where n_transactions > 0 should also have revenue > 0
        active = daily_df[daily_df["n_transactions"] > 0]
        assert (active["revenue"] > 0).all()

    def test_gaps_filled_with_zero(self, txn_df):
        """aggregate_daily should reindex to a complete date range filled with 0."""
        daily = aggregate_daily(txn_df)
        for product in daily["product"].unique():
            pdf = daily[daily["product"] == product].sort_values("date")
            all_dates = pd.date_range(pdf["date"].min(), pdf["date"].max(), freq="D")
            assert len(pdf) == len(all_dates), (
                f"Product '{product}' has gaps: {len(pdf)} rows vs {len(all_dates)} expected"
            )

    def test_only_forecast_products_retained(self):
        """Products not in FORECAST_PRODUCTS should be dropped."""
        txn = _make_transactions(products=["Latte", "NotAProduct"], n_days=5)
        daily = aggregate_daily(txn)
        assert "NotAProduct" not in daily["product"].values

    def test_revenue_matches_sum(self, txn_df):
        """Revenue column must equal the actual sum of money per (product, date)."""
        txn_df2 = txn_df.copy()
        txn_df2["date"] = txn_df2["datetime"].dt.normalize()
        expected = (
            txn_df2.groupby(["coffee_name", "date"])["money"]
            .sum()
            .reset_index()
        )
        daily = aggregate_daily(txn_df)
        for _, row in expected.iterrows():
            if row["coffee_name"] not in FORECAST_PRODUCTS:
                continue
            match = daily[
                (daily["product"] == row["coffee_name"]) & (daily["date"] == row["date"])
            ]
            if not match.empty:
                assert match["revenue"].values[0] == pytest.approx(row["money"], rel=1e-5)


# ---------------------------------------------------------------------------
# Tests — add_lag_features
# ---------------------------------------------------------------------------


class TestAddLagFeatures:
    def test_lag_columns_created(self, daily_df):
        result = add_lag_features(daily_df)
        assert {"lag_1", "lag_7", "lag_14"}.issubset(result.columns)

    def test_lag_1_correct_value(self, daily_df):
        """lag_1 for day T must equal revenue of day T-1 for the same product."""
        result = add_lag_features(daily_df).sort_values(["product", "date"])
        for product in result["product"].unique():
            pdf = result[result["product"] == product].sort_values("date").reset_index(drop=True)
            for i in range(1, len(pdf)):
                expected = pdf.loc[i - 1, "revenue"]
                actual = pdf.loc[i, "lag_1"]
                assert actual == pytest.approx(expected, abs=1e-9), (
                    f"lag_1 mismatch for {product} on {pdf.loc[i, 'date']}"
                )

    def test_no_future_leakage(self, daily_df):
        """No lag value should be computed using data from a future row."""
        result = add_lag_features(daily_df).sort_values(["product", "date"])
        for product in result["product"].unique():
            pdf = result[result["product"] == product].sort_values("date").reset_index(drop=True)
            # lag_7 on row i must equal revenue of row i-7 (or 0 for the first 7 rows)
            for lag, col in [(1, "lag_1"), (7, "lag_7"), (14, "lag_14")]:
                for i in range(lag, len(pdf)):
                    expected = pdf.loc[i - lag, "revenue"]
                    actual = pdf.loc[i, col]
                    assert actual == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests — add_rolling_features
# ---------------------------------------------------------------------------


class TestAddRollingFeatures:
    def test_rolling_columns_created(self, daily_df):
        result = add_rolling_features(daily_df)
        assert {"rolling_mean_7", "rolling_mean_14"}.issubset(result.columns)

    def test_rolling_mean_excludes_current_day(self, daily_df):
        """rolling_mean_7 must NOT include the current day's revenue (shift(1))."""
        result = add_rolling_features(daily_df).sort_values(["product", "date"])
        for product in result["product"].unique():
            pdf = result[result["product"] == product].sort_values("date").reset_index(drop=True)
            # For a day with a spike, rolling_mean_7 the next day should include it
            # but the same day's rolling_mean_7 must NOT include the current revenue
            if len(pdf) > 8:
                i = 8
                window = pdf.loc[max(0, i - 7): i - 1, "revenue"].values
                expected_mean = window.mean() if len(window) > 0 else 0.0
                actual = pdf.loc[i, "rolling_mean_7"]
                assert actual == pytest.approx(expected_mean, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests — add_calendar_features
# ---------------------------------------------------------------------------


class TestAddCalendarFeatures:
    def test_calendar_columns_created(self, daily_df):
        result = add_calendar_features(daily_df)
        assert {"day_of_week", "is_weekend", "month"}.issubset(result.columns)

    def test_day_of_week_range(self, daily_df):
        result = add_calendar_features(daily_df)
        assert result["day_of_week"].between(0, 6).all()

    def test_is_weekend_boolean_values(self, daily_df):
        result = add_calendar_features(daily_df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_weekend_matches_day_of_week(self, daily_df):
        result = add_calendar_features(daily_df)
        expected_weekend = (result["day_of_week"] >= 5).astype("int8")
        assert (result["is_weekend"] == expected_weekend).all()

    def test_month_range(self, daily_df):
        result = add_calendar_features(daily_df)
        assert result["month"].between(1, 12).all()


# ---------------------------------------------------------------------------
# Tests — build_features (end-to-end)
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_required_columns_present(self, txn_df):
        result = build_features(txn_df)
        required = {
            "product", "date", "revenue", "n_transactions",
            "lag_1", "lag_7", "lag_14",
            "rolling_mean_7", "rolling_mean_14",
            "day_of_week", "is_weekend", "month",
        }
        assert required.issubset(result.columns)

    def test_no_null_in_feature_columns(self, txn_df):
        result = build_features(txn_df)
        feature_cols = [
            "lag_1", "lag_7", "lag_14",
            "rolling_mean_7", "rolling_mean_14",
            "day_of_week", "is_weekend", "month",
        ]
        assert result[feature_cols].isna().sum().sum() == 0

    def test_rows_equal_products_times_days(self, txn_df):
        result = build_features(txn_df)
        n_products = result["product"].nunique()
        n_days = (result["date"].max() - result["date"].min()).days + 1
        assert len(result) == n_products * n_days
