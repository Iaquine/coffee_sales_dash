"""
rfm.py — Quintile-based RFM segmentation (classic marketing approach).

Provides ``score_rfm(df)`` which takes an RFM feature DataFrame and returns
a copy augmented with:
    r_quintile, f_quintile, m_quintile  (1–5 per dimension)
    rfm_score                           (sum of three quintiles, range 3–15)
    segment_name                        (named marketing segment)

Segment mapping (by rfm_score range)
-------------------------------------
11–15 → Champions     : highest recency, frequency, and monetary value
 8–10 → Loyal         : consistently purchasing, good monetary value
  6–7 → Potential     : moderately recent, moderate frequency
  4–5 → At Risk       : declining activity, previously active
  3   → Lost          : lowest recency, frequency, and monetary value

Notes
-----
- Quintiles are computed on the passed DataFrame (i.e. relative to the
  population supplied — typically the full RFM feature store).
- For Recency the rank is *inverted* before quintile assignment: a smaller
  recency_days value (more recent) maps to a higher quintile (5 = best).
"""

from __future__ import annotations

import pandas as pd


def _build_segment_lookup() -> dict[int, str]:
    """Pre-build a lookup dict {score: segment} for scores 3..15."""
    lookup: dict[int, str] = {}
    bounds = [
        (11, 15, "Champions"),
        (8,  10, "Loyal"),
        (6,   7, "Potential"),
        (4,   5, "At Risk"),
        (3,   3, "Lost"),
    ]
    for lo, hi, name in bounds:
        for s in range(lo, hi + 1):
            lookup[s] = name
    return lookup


# Built once at import time for O(1) lookups inside score_rfm
_SCORE_LOOKUP: dict[int, str] = _build_segment_lookup()


def score_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Assign quintile scores and a named segment to each customer.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``recency_days``, ``frequency``, ``monetary``.
        Typically the output of :func:`src.features.rfm_features.compute_rfm`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with five additional columns:
        ``r_quintile``, ``f_quintile``, ``m_quintile``,
        ``rfm_score``, ``segment_name``.
    """
    df = df.copy()

    # Recency: rank ascending=False so smallest recency_days (most recent) → highest rank → Q5
    df["r_quintile"] = pd.qcut(
        df["recency_days"].rank(method="first", ascending=False),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    # Frequency: rank ascending=True so highest frequency → highest quintile
    df["f_quintile"] = pd.qcut(
        df["frequency"].rank(method="first", ascending=True),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    # Monetary: rank ascending=True so highest spend → highest quintile
    df["m_quintile"] = pd.qcut(
        df["monetary"].rank(method="first", ascending=True),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    df["rfm_score"] = df["r_quintile"] + df["f_quintile"] + df["m_quintile"]
    df["segment_name"] = df["rfm_score"].map(_SCORE_LOOKUP)

    return df
