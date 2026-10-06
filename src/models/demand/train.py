"""
train.py — Champion model training for demand forecasting.

The champion is selected per the notebook evaluation (Section 6). XGBoost with
lag features is used as the champion (and guaranteed fallback if Prophet is
unavailable).

Public API
----------
train(product, df_train, params) -> tuple[model, dict]
    Train the champion model for a single product and return (fitted_model, metrics).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns used by the XGBoost champion
# ---------------------------------------------------------------------------

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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = _mape(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix from a demand-features DataFrame."""
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")
    return df[ALL_FEATURES].copy()


# ---------------------------------------------------------------------------
# XGBoost champion
# ---------------------------------------------------------------------------


def _train_xgboost(df_train: pd.DataFrame, params: dict[str, Any]):
    """Train an XGBoost regressor on the lag-feature matrix."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for the champion model") from exc

    X_train = _prepare_X(df_train)
    y_train = df_train["revenue"].values

    default_params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    default_params.update(params or {})

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    logger.info("XGBoost trained on %d rows for product (params=%s)", len(df_train), default_params)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    product: str,
    df_train: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train the demand-forecasting champion model for a single product.

    Parameters
    ----------
    product:
        Product name (used for logging only; filtering must be done by caller).
    df_train:
        Feature-store DataFrame filtered to the training period and product.
        Must contain columns in ``ALL_FEATURES`` plus ``revenue``.
    params:
        Optional hyperparameter overrides passed directly to the underlying
        XGBoost regressor.

    Returns
    -------
    model:
        Fitted XGBRegressor instance.
    metrics:
        Dict with ``mae``, ``rmse``, ``mape`` computed on the *training* set
        (informational; evaluation on the hold-out is done in evaluate.py).
    """
    params = params or {}
    logger.info("Training champion model for product='%s'", product)

    model = _train_xgboost(df_train, params)

    X_train = _prepare_X(df_train)
    y_train = df_train["revenue"].values
    y_pred_train = model.predict(X_train)
    metrics = _metrics(y_train, y_pred_train)

    logger.info(
        "product='%s' train metrics — MAE=%.2f RMSE=%.2f MAPE=%.2f%%",
        product,
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
    )
    return model, metrics


def predict(model: Any, df: pd.DataFrame) -> np.ndarray:
    """Generate revenue predictions for a prepared feature-store DataFrame.

    Parameters
    ----------
    model:
        Fitted model returned by :func:`train`.
    df:
        DataFrame containing the feature columns in ``ALL_FEATURES``.

    Returns
    -------
    np.ndarray
        Array of predicted revenue values (clipped at 0).
    """
    X = _prepare_X(df)
    preds = model.predict(X)
    return np.clip(preds, 0, None)
