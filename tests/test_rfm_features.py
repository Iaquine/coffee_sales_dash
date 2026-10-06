"""
test_rfm_features.py — Unit tests for src/features/rfm_features.py

Coverage:
  - compute_rfm: correct Recency, Frequency, Monetary values on synthetic data
  - normalisation range: recency_norm, frequency_norm, monetary_norm all in [0, 1]
  - anonymous customers excluded from the output
  - null card values excluded from the output
  - reference date = last observed date in dataset (reproducibility)
  - column names and DataFrame shape
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.rfm_features import compute_rfm, _minmax, _minmax_invert

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_txn() -> pd.DataFrame:
    """Minimal synthetic transactions with 3 identified customers and 1 anonymous."""
    ref_date = pd.Timestamp("2024-03-31")

    rows = [
        # card A: last purchase = ref_date-2, 5 txns, total = 50
        {"datetime": ref_date - pd.Timedelta(days=2),  "card": "CARD_A", "money": 20.0},
        {"datetime": ref_date - pd.Timedelta(days=5),  "card": "CARD_A", "money": 10.0},
        {"datetime": ref_date - pd.Timedelta(days=10), "card": "CARD_A", "money": 5.0},
        {"datetime": ref_date - pd.Timedelta(days=15), "card": "CARD_A", "money": 8.0},
        {"datetime": ref_date - pd.Timedelta(days=20), "card": "CARD_A", "money": 7.0},
        # card B: last purchase = ref_date-30, 2 txns, total = 30
        {"datetime": ref_date - pd.Timedelta(days=30), "card": "CARD_B", "money": 15.0},
        {"datetime": ref_date - pd.Timedelta(days=60), "card": "CARD_B", "money": 15.0},
        # card C: last purchase = ref_date-0 (same day), 1 txn, total = 100
        {"datetime": ref_date,                          "card": "CARD_C", "money": 100.0},
        # anonymous — must be excluded
        {"datetime": ref_date - pd.Timedelta(days=1), "card": None,        "money": 5.0},
        {"datetime": ref_date - pd.Timedelta(days=1), "card": "anonymous", "money": 5.0},
    ]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["cash_type"] = "card"
    df["coffee_name"] = "Latte"
    return df


# ---------------------------------------------------------------------------
# Tests — compute_rfm correctness
# ---------------------------------------------------------------------------


class TestComputeRfm:
    def test_output_columns(self, small_txn):
        rfm = compute_rfm(small_txn)
        expected = {
            "card", "recency_days", "frequency", "monetary",
            "recency_norm", "frequency_norm", "monetary_norm",
        }
        assert expected.issubset(rfm.columns)

    def test_anonymous_excluded(self, small_txn):
        rfm = compute_rfm(small_txn)
        assert "anonymous" not in rfm["card"].values

    def test_null_card_excluded(self, small_txn):
        rfm = compute_rfm(small_txn)
        assert rfm["card"].isna().sum() == 0

    def test_three_customers_returned(self, small_txn):
        rfm = compute_rfm(small_txn)
        assert len(rfm) == 3, f"Expected 3 identified customers, got {len(rfm)}"

    def test_recency_correct_for_card_c(self, small_txn):
        """CARD_C bought on the reference date → recency_days = 0."""
        rfm = compute_rfm(small_txn).set_index("card")
        assert rfm.loc["CARD_C", "recency_days"] == 0

    def test_recency_correct_for_card_a(self, small_txn):
        """CARD_A last purchase was 2 days before ref_date → recency_days = 2."""
        rfm = compute_rfm(small_txn).set_index("card")
        assert rfm.loc["CARD_A", "recency_days"] == 2

    def test_frequency_correct_for_card_a(self, small_txn):
        rfm = compute_rfm(small_txn).set_index("card")
        assert rfm.loc["CARD_A", "frequency"] == 5

    def test_monetary_correct_for_card_a(self, small_txn):
        rfm = compute_rfm(small_txn).set_index("card")
        assert rfm.loc["CARD_A", "monetary"] == pytest.approx(50.0, abs=1e-9)

    def test_monetary_correct_for_card_c(self, small_txn):
        rfm = compute_rfm(small_txn).set_index("card")
        assert rfm.loc["CARD_C", "monetary"] == pytest.approx(100.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests — normalisation in [0, 1]
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_recency_norm_in_0_1(self, small_txn):
        rfm = compute_rfm(small_txn)
        assert rfm["recency_norm"].between(0.0, 1.0).all(), (
            f"recency_norm out of [0,1]: {rfm['recency_norm'].tolist()}"
        )

    def test_frequency_norm_in_0_1(self, small_txn):
        rfm = compute_rfm(small_txn)
        assert rfm["frequency_norm"].between(0.0, 1.0).all()

    def test_monetary_norm_in_0_1(self, small_txn):
        rfm = compute_rfm(small_txn)
        assert rfm["monetary_norm"].between(0.0, 1.0).all()

    def test_most_recent_has_highest_recency_norm(self, small_txn):
        """CARD_C bought today → should have the highest recency_norm (most recent)."""
        rfm = compute_rfm(small_txn).set_index("card")
        assert rfm.loc["CARD_C", "recency_norm"] == pytest.approx(1.0, abs=1e-9)

    def test_least_frequent_has_lowest_frequency_norm(self, small_txn):
        """CARD_C bought only once → frequency_norm should be min (0.0)."""
        rfm = compute_rfm(small_txn).set_index("card")
        assert rfm.loc["CARD_C", "frequency_norm"] == pytest.approx(0.0, abs=1e-9)

    def test_highest_spender_has_highest_monetary_norm(self, small_txn):
        """CARD_C spent 100 (max) → monetary_norm should be 1.0."""
        rfm = compute_rfm(small_txn).set_index("card")
        assert rfm.loc["CARD_C", "monetary_norm"] == pytest.approx(1.0, abs=1e-9)

    def test_minmax_constant_series_returns_half(self):
        s = pd.Series([5.0, 5.0, 5.0])
        result = _minmax(s)
        assert (result == 0.5).all()

    def test_minmax_invert_reverses_order(self):
        """minmax_invert: the smallest raw value should map to norm = 1.0."""
        s = pd.Series([0.0, 5.0, 10.0])
        result = _minmax_invert(s)
        assert result.iloc[0] == pytest.approx(1.0, abs=1e-9)
        assert result.iloc[2] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests — reference date reproducibility
# ---------------------------------------------------------------------------


class TestReferenceDate:
    def test_reference_date_is_last_transaction(self, small_txn):
        """Reference date must equal the last observed datetime (floor to day)."""
        rfm = compute_rfm(small_txn)
        # CARD_C bought on 2024-03-31 (ref_date) → min recency_days should be 0
        assert rfm["recency_days"].min() == 0

    def test_result_stable_across_calls(self, small_txn):
        """compute_rfm must be deterministic for the same input."""
        rfm1 = compute_rfm(small_txn)
        rfm2 = compute_rfm(small_txn)
        pd.testing.assert_frame_equal(rfm1.sort_values("card").reset_index(drop=True),
                                      rfm2.sort_values("card").reset_index(drop=True))


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_customer_norms_are_half_or_one(self):
        """With a single customer the constant series returns 0.5 for each norm."""
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")],
            "card": ["ONLY_ONE", "ONLY_ONE"],
            "money": [10.0, 20.0],
        })
        rfm = compute_rfm(df)
        assert len(rfm) == 1
        assert rfm["recency_norm"].iloc[0] == pytest.approx(0.5, abs=1e-9)
        assert rfm["frequency_norm"].iloc[0] == pytest.approx(0.5, abs=1e-9)
        assert rfm["monetary_norm"].iloc[0] == pytest.approx(0.5, abs=1e-9)

    def test_anonymous_rows_stripped_from_mixed_df(self):
        """Anonymous rows in a mixed DataFrame are excluded; only identified customers remain."""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ],
            "card": ["REAL_CARD", "anonymous", None],
            "money": [10.0, 5.0, 5.0],
        })
        rfm = compute_rfm(df)
        assert len(rfm) == 1
        assert rfm["card"].iloc[0] == "REAL_CARD"
