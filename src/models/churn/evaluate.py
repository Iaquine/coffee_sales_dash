"""Churn model evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a binary churn classifier.

    Parameters
    ----------
    model : fitted classifier
        Any sklearn-compatible model that exposes ``predict_proba``.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True binary labels.
    threshold : float
        Decision threshold applied to probabilities to derive hard labels.

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``, ``roc_auc``, ``pr_auc``.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
    }
