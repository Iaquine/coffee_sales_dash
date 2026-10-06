"""
test_churn_features.py — Unit tests for src/features/churn_features.py

Coverage:
  - churn label is 1 for customers whose last purchase falls before the churn window
  - churn label is 0 for customers who purchased within the churn window
  - freq_last_4w is 0 when the customer has no recent purchases
  - avg_ticket_recent is 0 when the customer has no recent purchases
  - anonymous and null-card customers are excluded
  - column set matches the expected schema
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.churn_features import build_churn_features

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_transactions(
    ref_date: pd.Timestamp,
    customer_last_days: dict[str, int],
    anchor_card: str = "_ANCHOR_",
) -> pd.DataFrame:
    """Build a minimal transactions DataFrame.

    Parameters
    ----------
    ref_date : the maximum datetime in the dataset (used as reference)
    customer_last_days : {card: days_before_ref_date_of_last_purchase}
        For each card, create one transaction at (ref_date - days) and
        one earlier transaction 90 days before that.
    anchor_card : a sentinel card that always has a transaction ON ref_date so
        that ``df["datetime"].max()`` equals *ref_date* reliably.
    """
    rows = []

    # Anchor row: ensures ref_date is truly the maximum datetime in the dataset
    rows.append({"datetime": ref_date, "card": anchor_card, "money": 1.0, "coffee_name": "Latte"})
    rows.append({"datetime": ref_date - pd.Timedelta(days=120), "card": anchor_card,
                 "money": 1.0, "coffee_name": "Latte"})

    for card, days in customer_last_days.items():
        last_dt = ref_date - pd.Timedelta(days=days)
        rows.append({"datetime": last_dt, "card": card, "money": 10.0, "coffee_name": "Latte"})
        # Older transaction to give each customer >1 appearance
        earlier_dt = last_dt - pd.Timedelta(days=90)
        rows.append({"datetime": earlier_dt, "card": card, "money": 12.0, "coffee_name": "Espresso"})

    # Include anonymous / null rows (should be excluded by build_churn_features)
    rows.append({"datetime": ref_date - pd.Timedelta(days=5), "card": None, "money": 5.0, "coffee_name": "Latte"})
    rows.append({"datetime": ref_date - pd.Timedelta(days=5), "card": "anonymous", "money": 5.0, "coffee_name": "Latte"})

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["cash_type"] = "card"
    return df


def _make_rfm(
    cards: list[str],
    ref_date: pd.Timestamp,
    customer_last_days: dict[str, int],
    extra_card: str | None = "_ANCHOR_",
) -> pd.DataFrame:
    """Minimal RFM table for the given cards (plus optional anchor)."""
    all_cards = list(cards) + ([extra_card] if extra_card else [])
    # anchor always bought on ref_date → recency_days = 0
    merged_days = dict(customer_last_days)
    if extra_card:
        merged_days[extra_card] = 0
    rows = []
    for card in all_cards:
        days = merged_days[card]
        rows.append({
            "card": card,
            "recency_days": days,
            "frequency": 2,
            "monetary": 22.0,
            "recency_norm": max(0.0, 1.0 - days / 120),
            "frequency_norm": 0.5,
            "monetary_norm": 0.5,
        })
    return pd.DataFrame(rows)


def _make_segments(cards: list[str], extra_card: str | None = "_ANCHOR_") -> pd.DataFrame:
    all_cards = list(cards) + ([extra_card] if extra_card else [])
    return pd.DataFrame({
        "card": all_cards,
        "cluster_id": [0] * len(all_cards),
        "segment_name": ["Loyal"] * len(all_cards),
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REF_DATE = pd.Timestamp("2024-03-31")

# customer A: last purchase 5 days ago → active (within 90-day window)
# customer B: last purchase 100 days ago → churned (beyond 90-day window)
CUSTOMER_LAST_DAYS = {"CARD_A": 5, "CARD_B": 100}
CARDS = list(CUSTOMER_LAST_DAYS.keys())


@pytest.fixture()
def churn_features(tmp_path) -> pd.DataFrame:
    """Build churn features from synthetic data using monkeypatched file paths."""
    txn_df = _make_transactions(REF_DATE, CUSTOMER_LAST_DAYS)
    rfm_df = _make_rfm(CARDS, REF_DATE, CUSTOMER_LAST_DAYS)
    seg_df = _make_segments(CARDS)

    # Write temporary parquet files for rfm and segments
    rfm_path = tmp_path / "rfm_features.parquet"
    seg_path = tmp_path / "customer_segments.parquet"
    raw_path = tmp_path / "coffee_sales.csv"

    rfm_df.to_parquet(rfm_path, index=False)
    seg_df.to_parquet(seg_path, index=False)
    txn_df.to_csv(raw_path, index=False)

    return build_churn_features(
        raw_csv=raw_path,
        rfm_parquet=rfm_path,
        segments_parquet=seg_path,
        churn_window_days=90,
    )


# ---------------------------------------------------------------------------
# Tests — churn label
# ---------------------------------------------------------------------------


class TestChurnLabel:
    def test_active_customer_label_zero(self, churn_features):
        """CARD_A bought 5 days ago → churn = 0 (active)."""
        row = churn_features[churn_features["card"] == "CARD_A"]
        assert len(row) == 1
        assert row["churn"].iloc[0] == 0

    def test_churned_customer_label_one(self, churn_features):
        """CARD_B bought 100 days ago (> 90) → churn = 1."""
        row = churn_features[churn_features["card"] == "CARD_B"]
        assert len(row) == 1
        assert row["churn"].iloc[0] == 1

    def test_churn_is_binary(self, churn_features):
        assert set(churn_features["churn"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests — recent features are 0 when no recent purchases
# ---------------------------------------------------------------------------


class TestRecentFeatures:
    def test_freq_last_4w_zero_for_churned(self, churn_features):
        """CARD_B has no purchase in the last 4 weeks → freq_last_4w = 0."""
        row = churn_features[churn_features["card"] == "CARD_B"]
        assert row["freq_last_4w"].iloc[0] == 0

    def test_freq_last_8w_zero_for_churned(self, churn_features):
        row = churn_features[churn_features["card"] == "CARD_B"]
        assert row["freq_last_8w"].iloc[0] == 0

    def test_avg_ticket_recent_zero_for_churned(self, churn_features):
        row = churn_features[churn_features["card"] == "CARD_B"]
        assert row["avg_ticket_recent"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_freq_last_4w_positive_for_active(self, churn_features):
        """CARD_A bought 5 days ago → freq_last_4w >= 1."""
        row = churn_features[churn_features["card"] == "CARD_A"]
        assert row["freq_last_4w"].iloc[0] >= 1


# ---------------------------------------------------------------------------
# Tests — anonymous customers excluded
# ---------------------------------------------------------------------------


class TestAnonymousExcluded:
    def test_null_card_not_in_output(self, churn_features):
        assert churn_features["card"].isna().sum() == 0

    def test_anonymous_card_not_in_output(self, churn_features):
        assert "anonymous" not in churn_features["card"].values

    def test_only_identified_customers_present(self, churn_features):
        # _ANCHOR_ card is included to anchor ref_date; anonymous/null are excluded
        assert {"CARD_A", "CARD_B"}.issubset(churn_features["card"].unique())
        assert "anonymous" not in churn_features["card"].values
        assert churn_features["card"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Tests — column schema
# ---------------------------------------------------------------------------


class TestColumnSchema:
    def test_required_columns_present(self, churn_features):
        required = {
            "card", "churn",
            "recency_days", "frequency", "monetary",
            "recency_norm", "frequency_norm", "monetary_norm",
            "cluster_id", "segment_name",
            "freq_last_4w", "freq_last_8w",
            "avg_ticket_recent", "n_distinct_products",
            "days_since_first_purchase",
        }
        assert required.issubset(churn_features.columns), (
            f"Missing columns: {required - set(churn_features.columns)}"
        )

    def test_n_distinct_products_positive(self, churn_features):
        assert (churn_features["n_distinct_products"] >= 1).all()

    def test_days_since_first_purchase_non_negative(self, churn_features):
        assert (churn_features["days_since_first_purchase"] >= 0).all()


# ---------------------------------------------------------------------------
# Tests — churn window boundary
# ---------------------------------------------------------------------------


class TestChurnWindowBoundary:
    def test_boundary_active_at_exactly_window_days(self, tmp_path):
        """A customer whose last purchase is exactly churn_window_days ago is active (not churned)."""
        window = 60
        ref = pd.Timestamp("2024-03-31")
        # last purchase exactly at the cutoff boundary
        customer_days = {"CARD_X": window}

        txn = _make_transactions(ref, customer_days)
        rfm = _make_rfm(["CARD_X"], ref, customer_days)
        seg = _make_segments(["CARD_X"])

        raw_path = tmp_path / "coffee_sales2.csv"
        rfm_path = tmp_path / "rfm2.parquet"
        seg_path = tmp_path / "seg2.parquet"

        txn.to_csv(raw_path, index=False)
        rfm.to_parquet(rfm_path, index=False)
        seg.to_parquet(seg_path, index=False)

        result = build_churn_features(
            raw_csv=raw_path,
            rfm_parquet=rfm_path,
            segments_parquet=seg_path,
            churn_window_days=window,
        )
        row = result[result["card"] == "CARD_X"]
        # last purchase NOT strictly before the cutoff → active
        assert row["churn"].iloc[0] == 0

    def test_one_day_past_window_is_churned(self, tmp_path):
        """A customer whose last purchase is churn_window_days + 1 ago is churned."""
        window = 60
        ref = pd.Timestamp("2024-03-31")
        customer_days = {"CARD_Y": window + 1}

        txn = _make_transactions(ref, customer_days)
        rfm = _make_rfm(["CARD_Y"], ref, customer_days)
        seg = _make_segments(["CARD_Y"])

        raw_path = tmp_path / "coffee_sales3.csv"
        rfm_path = tmp_path / "rfm3.parquet"
        seg_path = tmp_path / "seg3.parquet"

        txn.to_csv(raw_path, index=False)
        rfm.to_parquet(rfm_path, index=False)
        seg.to_parquet(seg_path, index=False)

        result = build_churn_features(
            raw_csv=raw_path,
            rfm_parquet=rfm_path,
            segments_parquet=seg_path,
            churn_window_days=window,
        )
        row = result[result["card"] == "CARD_Y"]
        assert row["churn"].iloc[0] == 1
