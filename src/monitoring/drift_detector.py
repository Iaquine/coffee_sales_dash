"""
drift_detector.py — Data drift and concept drift detection.

Public API
----------
compute_psi(expected, actual, bins=10) -> float
    Population Stability Index between two distributions.

ks_test(expected, actual) -> dict
    Kolmogorov–Smirnov test: {'statistic': float, 'p_value': float}.

detect_data_drift(df_reference, df_current, features) -> dict
    PSI + KS per feature; sets drift_detected=True if any PSI > 0.2.

detect_concept_drift(metrics_history, metric_key, threshold_pct=0.20) -> dict
    Compare first-quarter baseline vs latest value of a metric series.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PSI_STABLE = 0.1
PSI_CRITICAL = 0.2


# ---------------------------------------------------------------------------
# Low-level statistics
# ---------------------------------------------------------------------------


def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI = Σ (actual% - expected%) * ln(actual% / expected%)

    Returns
    -------
    float
        PSI value (0 = identical, > 0.2 = significant shift).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Build bin edges from the combined range
    combined = np.concatenate([expected, actual])
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    # Ensure unique breakpoints to avoid zero-width bins
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to proportions; add small epsilon to avoid log(0)
    eps = 1e-6
    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def ks_test(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Two-sample Kolmogorov–Smirnov test.

    Returns
    -------
    dict with keys 'statistic' and 'p_value'.
    """
    stat, p_value = stats.ks_2samp(
        np.asarray(expected, dtype=float),
        np.asarray(actual, dtype=float),
    )
    return {"statistic": float(stat), "p_value": float(p_value)}


# ---------------------------------------------------------------------------
# Data drift
# ---------------------------------------------------------------------------


def detect_data_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    features: list[str],
    psi_bins: int = 10,
    psi_threshold: float = PSI_CRITICAL,
) -> dict[str, Any]:
    """Detect data drift for a list of numeric features.

    Parameters
    ----------
    df_reference : pd.DataFrame
        Reference (baseline) dataset.
    df_current : pd.DataFrame
        Current production dataset.
    features : list[str]
        Column names to evaluate.
    psi_bins : int
        Number of bins for PSI computation.
    psi_threshold : float
        PSI value above which drift is flagged.

    Returns
    -------
    dict with keys:
        - 'features': per-feature dict of {'psi', 'ks_statistic', 'ks_p_value'}
        - 'drift_detected': True if any feature PSI exceeds the threshold
    """
    results: dict[str, Any] = {"features": {}, "drift_detected": False}

    for feat in features:
        if feat not in df_reference.columns or feat not in df_current.columns:
            logger.warning("Feature '%s' missing from one of the dataframes — skipped.", feat)
            continue

        ref_vals = df_reference[feat].dropna().values
        cur_vals = df_current[feat].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            logger.warning("Feature '%s' has no valid values — skipped.", feat)
            continue

        psi_val = compute_psi(ref_vals, cur_vals, bins=psi_bins)
        ks_result = ks_test(ref_vals, cur_vals)

        results["features"][feat] = {
            "psi": round(psi_val, 4),
            "ks_statistic": round(ks_result["statistic"], 4),
            "ks_p_value": round(ks_result["p_value"], 4),
        }

        if psi_val > psi_threshold:
            results["drift_detected"] = True
            logger.warning(
                "Data drift detected on feature '%s': PSI=%.4f (threshold=%.2f)",
                feat,
                psi_val,
                psi_threshold,
            )

    return results


# ---------------------------------------------------------------------------
# Concept drift
# ---------------------------------------------------------------------------


def detect_concept_drift(
    metrics_history: list[dict],
    metric_key: str,
    threshold_pct: float = 0.20,
) -> dict[str, Any]:
    """Detect concept drift by comparing a baseline window vs the latest value.

    The baseline is computed as the mean of the first quarter of the history.
    The current value is the last entry in the history.

    Parameters
    ----------
    metrics_history : list[dict]
        Ordered list of metric snapshots, each containing at least ``metric_key``.
    metric_key : str
        Name of the metric to track (e.g. 'mape', 'pr_auc').
    threshold_pct : float
        Relative degradation that triggers a drift flag (default 20 %).

    Returns
    -------
    dict with keys: 'drift_detected', 'baseline_value', 'current_value',
    'degradation_pct'.
    """
    if not metrics_history:
        return {
            "drift_detected": False,
            "baseline_value": None,
            "current_value": None,
            "degradation_pct": 0.0,
        }

    values = [m[metric_key] for m in metrics_history if metric_key in m]
    if not values:
        raise KeyError(f"metric_key '{metric_key}' not found in metrics_history entries")

    n_baseline = max(1, len(values) // 4)
    baseline_value = float(np.mean(values[:n_baseline]))
    current_value = float(values[-1])

    # For error metrics (MAPE): degradation = increase; for score metrics (AUC):
    # degradation = decrease.  We express degradation as a relative change where
    # positive = worse.
    if baseline_value == 0:
        degradation_pct = 0.0
    else:
        degradation_pct = (current_value - baseline_value) / abs(baseline_value)

    drift_detected = abs(degradation_pct) > threshold_pct

    if drift_detected:
        logger.warning(
            "Concept drift detected on '%s': baseline=%.4f, current=%.4f, "
            "degradation=%.1f%%",
            metric_key,
            baseline_value,
            current_value,
            degradation_pct * 100,
        )

    return {
        "drift_detected": drift_detected,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "degradation_pct": round(degradation_pct, 4),
    }
