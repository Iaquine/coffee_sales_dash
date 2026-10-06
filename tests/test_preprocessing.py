"""
test_preprocessing.py — Unit tests for src/data/preprocessing.py.

All tests operate on synthetic DataFrames constructed inline — no dependency
on the real dataset file.
"""

from __future__ import annotations

import math
import pandas as pd
import pytest

from src.data.preprocessing import (
    extract_temporal_features,
    normalise_card,
    parse_datetime,
    preprocess,
    remove_invalid_rows,
    remove_monetary_outliers,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_base_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid raw DataFrame with 5 rows.

    Keyword overrides replace specific column values for targeted testing.
    """
    data = {
        "datetime": [
            "2024-01-15 07:30:00",  # morning, Mon
            "2024-01-15 13:00:00",  # afternoon, Mon
            "2024-01-16 18:45:00",  # evening, Tue
            "2024-01-20 23:10:00",  # night, Sat (weekend)
            "2024-01-21 03:05:00",  # night, Sun (weekend)
        ],
        "cash_type": ["card", "cash", "card", "card", "cash"],
        "card": ["CARD_A", "CARD_B", None, "CARD_D", "CARD_E"],
        "money": [3.50, 4.25, 5.00, 6.75, 2.80],
        "coffee_name": ["Latte", "Espresso", "Cappuccino", "Americano", "Latte"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Step 1 — parse_datetime
# ---------------------------------------------------------------------------


class TestParseDatetime:
    def test_converts_string_to_datetime(self):
        df = make_base_df()
        result = parse_datetime(df)
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_already_datetime_passthrough(self):
        df = make_base_df()
        df["datetime"] = pd.to_datetime(df["datetime"])
        result = parse_datetime(df)
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_drops_unparseable_rows(self):
        df = make_base_df(datetime=["not-a-date", "2024-01-15 07:30:00", "2024-01-16 18:45:00", "2024-01-20 23:10:00", "2024-01-21 03:05:00"])
        result = parse_datetime(df)
        assert len(result) == 4

    def test_does_not_mutate_input(self):
        df = make_base_df()
        original_dtype = df["datetime"].dtype
        parse_datetime(df)
        assert df["datetime"].dtype == original_dtype


# ---------------------------------------------------------------------------
# Step 2 — extract_temporal_features
# ---------------------------------------------------------------------------


class TestExtractTemporalFeatures:
    @pytest.fixture
    def parsed_df(self):
        df = parse_datetime(make_base_df())
        return extract_temporal_features(df)

    def test_hour_range(self, parsed_df):
        assert parsed_df["hour"].between(0, 23).all()

    def test_day_of_week_range(self, parsed_df):
        assert parsed_df["day_of_week"].between(0, 6).all()

    def test_week_of_year_range(self, parsed_df):
        assert parsed_df["week_of_year"].between(1, 53).all()

    def test_month_range(self, parsed_df):
        assert parsed_df["month"].between(1, 12).all()

    def test_is_weekend_type(self, parsed_df):
        assert parsed_df["is_weekend"].dtype == bool

    def test_saturday_is_weekend(self):
        # 2024-01-20 is a Saturday
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-20 10:00:00"]),
            "cash_type": ["card"],
            "card": ["X"],
            "money": [3.0],
            "coffee_name": ["Latte"],
        })
        result = extract_temporal_features(df)
        assert result["is_weekend"].iloc[0] == True  # noqa: E712

    def test_monday_is_not_weekend(self):
        # 2024-01-15 is a Monday
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-15 10:00:00"]),
            "cash_type": ["card"],
            "card": ["X"],
            "money": [3.0],
            "coffee_name": ["Latte"],
        })
        result = extract_temporal_features(df)
        assert result["is_weekend"].iloc[0] == False  # noqa: E712

    @pytest.mark.parametrize("hour,expected_bucket", [
        (0, "night"),
        (3, "night"),
        (5, "night"),
        (6, "morning"),
        (11, "morning"),
        (12, "afternoon"),
        (17, "afternoon"),
        (18, "evening"),
        (20, "evening"),
        (21, "night"),
        (23, "night"),
    ])
    def test_time_of_day_bucket_mapping(self, hour, expected_bucket):
        df = pd.DataFrame({
            "datetime": pd.to_datetime([f"2024-01-15 {hour:02d}:00:00"]),
            "cash_type": ["card"],
            "card": ["X"],
            "money": [3.0],
            "coffee_name": ["Latte"],
        })
        result = extract_temporal_features(df)
        assert result["time_of_day_bucket"].iloc[0] == expected_bucket

    def test_all_expected_columns_present(self, parsed_df):
        expected = {"hour", "day_of_week", "week_of_year", "month", "year", "is_weekend", "time_of_day_bucket"}
        assert expected.issubset(set(parsed_df.columns))


# ---------------------------------------------------------------------------
# Step 3 — remove_invalid_rows
# ---------------------------------------------------------------------------


class TestRemoveInvalidRows:
    def test_removes_zero_money(self):
        df = parse_datetime(make_base_df())
        df.loc[0, "money"] = 0.0
        result = remove_invalid_rows(df)
        assert len(result) == 4

    def test_removes_negative_money(self):
        df = parse_datetime(make_base_df())
        df.loc[1, "money"] = -1.5
        result = remove_invalid_rows(df)
        assert len(result) == 4

    def test_removes_null_coffee_name(self):
        df = parse_datetime(make_base_df())
        df.loc[2, "coffee_name"] = None
        result = remove_invalid_rows(df)
        assert len(result) == 4

    def test_removes_both_conditions_independently(self):
        df = parse_datetime(make_base_df())
        df.loc[0, "money"] = 0.0
        df.loc[1, "coffee_name"] = None
        result = remove_invalid_rows(df)
        assert len(result) == 3

    def test_preserves_valid_rows(self):
        df = parse_datetime(make_base_df())
        result = remove_invalid_rows(df)
        assert len(result) == 5

    def test_does_not_mutate_input(self):
        df = parse_datetime(make_base_df())
        df.loc[0, "money"] = 0.0
        original_len = len(df)
        remove_invalid_rows(df)
        assert len(df) == original_len


# ---------------------------------------------------------------------------
# Step 4 — remove_monetary_outliers
# ---------------------------------------------------------------------------


class TestRemoveMonetaryOutliers:
    def test_removes_high_outlier(self):
        # Build a DataFrame where one value is far beyond the IQR upper fence
        base_values = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-15 10:00:00"] * 11),
            "cash_type": ["card"] * 11,
            "card": ["X"] * 11,
            "money": base_values + [1000.0],
            "coffee_name": ["Latte"] * 11,
        })
        result = remove_monetary_outliers(df)
        assert 1000.0 not in result["money"].values
        assert len(result) == 10

    def test_normal_distribution_has_no_outliers(self):
        money = [3.0, 3.5, 4.0, 4.0, 4.5, 4.5, 5.0, 5.0, 5.5, 6.0]
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-15 10:00:00"] * len(money)),
            "cash_type": ["card"] * len(money),
            "card": ["X"] * len(money),
            "money": money,
            "coffee_name": ["Latte"] * len(money),
        })
        result = remove_monetary_outliers(df)
        assert len(result) == len(money)

    def test_return_type_is_dataframe(self):
        df = parse_datetime(make_base_df())
        result = remove_monetary_outliers(df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Step 5 — normalise_card
# ---------------------------------------------------------------------------


class TestNormaliseCard:
    def test_fills_null_card_with_anonymous(self):
        df = make_base_df()  # row 2 has card=None
        result = normalise_card(df)
        assert "anonymous" in result["card"].values
        assert result["card"].isna().sum() == 0

    def test_non_null_cards_unchanged(self):
        df = make_base_df()
        result = normalise_card(df)
        assert result["card"].iloc[0] == "CARD_A"
        assert result["card"].iloc[1] == "CARD_B"

    def test_all_null_cards(self):
        df = make_base_df(card=[None, None, None, None, None])
        result = normalise_card(df)
        assert (result["card"] == "anonymous").all()

    def test_no_null_cards_unchanged(self):
        df = make_base_df(card=["A", "B", "C", "D", "E"])
        result = normalise_card(df)
        assert result["card"].tolist() == ["A", "B", "C", "D", "E"]

    def test_does_not_mutate_input(self):
        df = make_base_df()  # row 2 has card=None
        normalise_card(df)
        assert pd.isna(df["card"].iloc[2])


# ---------------------------------------------------------------------------
# Full pipeline — preprocess()
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_output_has_all_temporal_columns(self):
        df = make_base_df()
        result = preprocess(df)
        expected = {"hour", "day_of_week", "week_of_year", "month", "year", "is_weekend", "time_of_day_bucket"}
        assert expected.issubset(set(result.columns))

    def test_no_null_card_in_output(self):
        df = make_base_df()
        result = preprocess(df)
        assert result["card"].isna().sum() == 0

    def test_no_zero_or_negative_money(self):
        df = make_base_df()
        df.loc[0, "money"] = 0.0
        result = preprocess(df)
        assert (result["money"] > 0).all()

    def test_no_null_coffee_name_in_output(self):
        df = make_base_df()
        df.loc[1, "coffee_name"] = None
        result = preprocess(df)
        assert result["coffee_name"].isna().sum() == 0

    def test_datetime_is_timestamp_dtype(self):
        df = make_base_df()
        result = preprocess(df)
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_outlier_removed_end_to_end(self):
        money = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 1000.0]
        df = pd.DataFrame({
            "datetime": ["2024-01-15 10:00:00"] * len(money),
            "cash_type": ["card"] * len(money),
            "card": ["X"] * len(money),
            "money": money,
            "coffee_name": ["Latte"] * len(money),
        })
        result = preprocess(df)
        assert 1000.0 not in result["money"].values

    def test_pipeline_does_not_mutate_input(self):
        df = make_base_df()
        original_datetime_dtype = df["datetime"].dtype
        preprocess(df)
        assert df["datetime"].dtype == original_datetime_dtype
        assert pd.isna(df["card"].iloc[2])
