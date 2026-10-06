"""XGBoost churn classifier — champion model training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def train(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    params: dict | None = None,
    random_state: int = 42,
) -> XGBClassifier:
    """Train the XGBoost churn champion model.

    Class imbalance is handled via ``scale_pos_weight`` computed automatically
    from ``y_train`` if not supplied in ``params``.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Binary target (1 = churned, 0 = active).
    params : dict, optional
        XGBoost hyperparameter overrides.  Any key not supplied falls back to
        the sensible defaults defined inside this function.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    XGBClassifier
        Fitted XGBoost classifier.
    """
    y_arr = np.asarray(y_train)
    n_neg = int((y_arr == 0).sum())
    n_pos = int((y_arr == 1).sum())
    default_spw = n_neg / max(n_pos, 1)

    defaults: dict = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": default_spw,
        "eval_metric": "aucpr",
        "random_state": random_state,
        "verbosity": 0,
        "n_jobs": -1,
    }

    if params:
        # Override scale_pos_weight only when explicitly provided
        if "scale_pos_weight" not in params:
            params["scale_pos_weight"] = default_spw
        defaults.update(params)

    model = XGBClassifier(**defaults)
    model.fit(X_train, y_train)
    return model
