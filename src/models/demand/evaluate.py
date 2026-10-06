"""
evaluate.py — Hold-out evaluation for demand forecasting models.

Public API
----------
evaluate(model, df_test, product) -> dict[str, float]
    Evaluate a fitted model on a hold-out DataFrame and return MAE, MAPE, RMSE.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Feature columns (mirrors train.py to avoid import coupling)
LAG_FEATURES = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14"]
CALENDAR_FEATURES = ["day_of_week", "is_weekend", "month"]
ALL_FEATURES = LAG_FEATURES + CALENDAR_FEATURES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, guarded against zero denominators."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    model: Any,
    df_test: pd.DataFrame,
    product: str,
) -> dict[str, float]:
    """Evaluate a fitted demand forecasting model on the hold-out set.

    Parameters
    ----------
    model:
        Fitted model with a ``predict(X)`` method (e.g., XGBRegressor).
    df_test:
        Hold-out feature-store DataFrame for a single product. Must contain
        the feature columns and a ``revenue`` column as ground truth.
    product:
        Product name used only for logging.

    Returns
    -------
    dict with keys ``mae``, ``rmse``, ``mape`` (float values).
    """
    missing = [c for c in ALL_FEATURES if c not in df_test.columns]
    if missing:
        raise ValueError(f"df_test is missing columns: {missing}")
    if "revenue" not in df_test.columns:
        raise ValueError("df_test must contain a 'revenue' column as ground truth")

    X_test = df_test[ALL_FEATURES]
    y_true = df_test["revenue"].values
    y_pred = np.clip(model.predict(X_test), 0, None)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = _mape(y_true, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "mape": mape}
    logger.info(
        "evaluate product='%s' — MAE=%.2f RMSE=%.2f MAPE=%.2f%%",
        product,
        mae,
        rmse,
        mape,
    )
    return metrics


def evaluate_prophet(model: Any, df_test: pd.DataFrame, product: str) -> dict[str, float]:
    """Evaluate a fitted Prophet model on the hold-out set.

    Parameters
    ----------
    model:
        Fitted Prophet instance.
    df_test:
        DataFrame with columns ``date`` and ``revenue``.
    product:
        Product name for logging.

    Returns
    -------
    dict with keys ``mae``, ``rmse``, ``mape``.
    """
    future = pd.DataFrame({"ds": df_test["date"].values})
    forecast = model.predict(future)
    y_pred = np.clip(forecast["yhat"].values, 0, None)
    y_true = df_test["revenue"].values

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = _mape(y_true, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "mape": mape}
    logger.info(
        "evaluate_prophet product='%s' — MAE=%.2f RMSE=%.2f MAPE=%.2f%%",
        product,
        mae,
        rmse,
        mape,
    )
    return metrics
