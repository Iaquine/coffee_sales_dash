"""Baseline churn classifier — Logistic Regression."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def train_logistic(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Train a Logistic Regression baseline for churn classification.

    Uses ``class_weight='balanced'`` to handle class imbalance and wraps the
    model in a ``StandardScaler`` pipeline so callers need not pre-scale data.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Binary target (1 = churned, 0 = active).
    max_iter : int
        Maximum solver iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline (scaler + logistic regression).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
