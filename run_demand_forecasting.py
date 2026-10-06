#!/usr/bin/env python
"""
run_demand_forecasting.py — Executes all Sub-Tarefa 4 steps directly (no notebook kernel overhead).

This script mirrors notebooks/02_demand_forecasting.ipynb section by section and produces
identical MLflow runs + demand_forecast.parquet output.
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent
FEATURES_PATH   = PROJECT_ROOT / "data" / "features" / "demand_features.parquet"
PREDICTIONS_PATH= PROJECT_ROOT / "data" / "predictions" / "demand_forecast.parquet"
MLRUNS_DIR      = PROJECT_ROOT / "mlruns"
MLRUNS_DIR.mkdir(exist_ok=True)
PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from src.features.demand_features import FORECAST_PRODUCTS
from src.features.demand_features import run as _feat_run  # noqa: F401 (used below)
from src.models.demand.evaluate import _mape, evaluate
from src.models.demand.train import train, ALL_FEATURES

# ── MLflow ──────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("demand_forecasting")

HOLD_OUT_DAYS = 30
N_TRIALS      = 20
SEED          = 42

# ── import demand_features ──────────────────────────────────────────────────
# ===========================================================================
# Section 1 — Feature store
# ===========================================================================
print("=" * 60)
print("Section 1 — Building feature store")
if not FEATURES_PATH.exists():
    _feat_run(output_path=FEATURES_PATH)
else:
    print(f"  Reusing existing feature store at {FEATURES_PATH}")

df_feat = pd.read_parquet(FEATURES_PATH, engine="pyarrow")
df_feat["date"] = pd.to_datetime(df_feat["date"])
print(f"  {len(df_feat):,} rows | {df_feat['product'].nunique()} products | "
      f"{df_feat['date'].min().date()} → {df_feat['date'].max().date()}")

# ===========================================================================
# Section 2 — Train / test split
# ===========================================================================
print("\n" + "=" * 60)
print("Section 2 — Train/test split")
max_date    = df_feat["date"].max()
cutoff_date = max_date - pd.Timedelta(days=HOLD_OUT_DAYS - 1)
df_train_all = df_feat[df_feat["date"] < cutoff_date].copy()
df_test_all  = df_feat[df_feat["date"] >= cutoff_date].copy()
print(f"  Cutoff: {cutoff_date.date()}  | Train: {len(df_train_all):,}  Test: {len(df_test_all):,}")

# ===========================================================================
# Section 3 — Seasonal Naive baseline
# ===========================================================================
print("\n" + "=" * 60)
print("Section 3 — Seasonal Naive baseline")

def seasonal_naive_predict(df_tr, df_te):
    predictions = []
    train_sorted = df_tr.sort_values("date")
    for _, row in df_te.iterrows():
        target_dow = row["date"].dayofweek
        same_dow = train_sorted[train_sorted["date"].dt.dayofweek == target_dow]
        predictions.append(same_dow.iloc[-1]["revenue"] if len(same_dow) else train_sorted["revenue"].mean())
    return np.array(predictions)

naive_results = {}
with mlflow.start_run(run_name="seasonal_naive_baseline"):
    mlflow.set_tag("model_type", "seasonal_naive")
    all_mape, all_mae, all_rmse = [], [], []
    for product in FORECAST_PRODUCTS:
        df_tr = df_train_all[df_train_all["product"] == product]
        df_te = df_test_all[df_test_all["product"] == product]
        y_pred = seasonal_naive_predict(df_tr, df_te)
        y_true = df_te["revenue"].values
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = _mape(y_true, y_pred)
        naive_results[product] = {"mae": mae, "rmse": rmse, "mape": mape}
        all_mape.append(mape); all_mae.append(mae); all_rmse.append(rmse)
        p_tag = product.replace(" ", "_")
        mlflow.log_metrics({f"{p_tag}_mae": mae, f"{p_tag}_rmse": rmse, f"{p_tag}_mape": mape})
        print(f"  {product:<25} MAE={mae:6.1f}  RMSE={rmse:6.1f}  MAPE={mape:5.1f}%")
    avg_mape = float(np.nanmean(all_mape))
    mlflow.log_metrics({"avg_mape": avg_mape, "avg_mae": float(np.nanmean(all_mae)), "avg_rmse": float(np.nanmean(all_rmse))})
    print(f"  Seasonal Naive avg MAPE={avg_mape:.1f}%")

# ===========================================================================
# Section 4 — Prophet
# ===========================================================================
print("\n" + "=" * 60)
print("Section 4 — Prophet")

PROPHET_AVAILABLE = False
prophet_results   = {}
try:
    import logging as _log
    from prophet import Prophet
    _log.getLogger("prophet").setLevel(_log.WARNING)
    _log.getLogger("cmdstanpy").setLevel(_log.WARNING)
    PROPHET_AVAILABLE = True
    print("  Prophet available — training per-product models")
except ImportError:
    print("  ⚠  Prophet not installed — skipping.")

if PROPHET_AVAILABLE:
    for product in FORECAST_PRODUCTS:
        df_tr = df_train_all[df_train_all["product"] == product][["date", "revenue"]].rename(columns={"date": "ds", "revenue": "y"})
        df_te = df_test_all[df_test_all["product"] == product][["date", "revenue"]].copy()
        with mlflow.start_run(run_name=f"prophet_{product.replace(' ','_')}"):
            mlflow.set_tag("model_type", "prophet")
            mlflow.set_tag("product", product)
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                        changepoint_prior_scale=0.05, seasonality_prior_scale=10, interval_width=0.95)
            m.fit(df_tr)
            future  = pd.DataFrame({"ds": df_te["date"].values})
            fc      = m.predict(future)
            y_pred  = np.clip(fc["yhat"].values, 0, None)
            y_true  = df_te["revenue"].values
            mae     = float(mean_absolute_error(y_true, y_pred))
            rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mape    = _mape(y_true, y_pred)
            prophet_results[product] = {"mae": mae, "rmse": rmse, "mape": mape}
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
            mlflow.log_params({"weekly_seasonality": True, "yearly_seasonality": True, "changepoint_prior_scale": 0.05})
            print(f"  {product:<25} MAE={mae:6.1f}  RMSE={rmse:6.1f}  MAPE={mape:5.1f}%")
    avg_mape_p = float(np.nanmean([v["mape"] for v in prophet_results.values()]))
    print(f"  Prophet avg MAPE={avg_mape_p:.1f}%")

# ===========================================================================
# Section 5 — XGBoost lag features
# ===========================================================================
print("\n" + "=" * 60)
print("Section 5 — XGBoost lag features")

import xgboost as xgb

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": SEED, "n_jobs": -1, "verbosity": 0,
}

xgb_results, xgb_models = {}, {}
for product in FORECAST_PRODUCTS:
    df_tr = df_train_all[df_train_all["product"] == product].copy()
    df_te = df_test_all[df_test_all["product"] == product].copy()
    with mlflow.start_run(run_name=f"xgboost_lag_{product.replace(' ','_')}"):
        mlflow.set_tag("model_type", "xgboost_lag")
        mlflow.set_tag("product", product)
        mlflow.log_params(DEFAULT_XGB_PARAMS)
        model, train_metrics = train(product, df_tr, DEFAULT_XGB_PARAMS.copy())
        test_metrics = evaluate(model, df_te, product)
        mlflow.log_metrics({"mae": test_metrics["mae"], "rmse": test_metrics["rmse"], "mape": test_metrics["mape"],
                            "train_mae": train_metrics["mae"], "train_rmse": train_metrics["rmse"]})
        mlflow.xgboost.log_model(model, artifact_path="model")
        xgb_results[product] = test_metrics
        xgb_models[product]  = model
    print(f"  {product:<25} MAE={test_metrics['mae']:6.1f}  RMSE={test_metrics['rmse']:6.1f}  MAPE={test_metrics['mape']:5.1f}%")

avg_mape_xgb = float(np.nanmean([v["mape"] for v in xgb_results.values()]))
print(f"  XGBoost avg MAPE={avg_mape_xgb:.1f}%")

# ===========================================================================
# Section 6 — Champion selection
# ===========================================================================
print("\n" + "=" * 60)
print("Section 6 — Champion selection")

mape_avgs = {"Seasonal Naive": float(np.nanmean([naive_results.get(p, {}).get("mape", float("nan")) for p in FORECAST_PRODUCTS]))}
if PROPHET_AVAILABLE and prophet_results:
    mape_avgs["Prophet"] = float(np.nanmean([prophet_results.get(p, {}).get("mape", float("nan")) for p in FORECAST_PRODUCTS]))
mape_avgs["XGBoost Lag"] = avg_mape_xgb

CHAMPION_NAME = min(mape_avgs, key=mape_avgs.get)
print("\n  Model Comparison — avg MAPE (%):")
for name, val in sorted(mape_avgs.items(), key=lambda x: x[1]):
    marker = " ← CHAMPION" if name == CHAMPION_NAME else ""
    print(f"    {name:<20}  avg MAPE={val:.1f}%{marker}")

# ===========================================================================
# Section 7 — Optuna tuning
# ===========================================================================
print("\n" + "=" * 60)
print(f"Section 7 — Optuna tuning ({N_TRIALS} trials per product)")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

best_params_per_product, tuned_models, tuned_results = {}, {}, {}

def make_objective(df_tr, df_te):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": SEED, "n_jobs": -1, "verbosity": 0,
        }
        X_tr = df_tr[ALL_FEATURES]; y_tr = df_tr["revenue"].values
        X_te = df_te[ALL_FEATURES]; y_te = df_te["revenue"].values
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr)
        y_pred = np.clip(m.predict(X_te), 0, None)
        return _mape(y_te, y_pred)
    return objective

for product in FORECAST_PRODUCTS:
    print(f"  Tuning {product}...", end=" ", flush=True)
    df_tr = df_train_all[df_train_all["product"] == product].copy()
    df_te = df_test_all[df_test_all["product"] == product].copy()
    study = optuna.create_study(direction="minimize", study_name=f"xgb_{product}",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_objective(df_tr, df_te), n_trials=N_TRIALS, show_progress_bar=False)
    best_p = {**study.best_params, "random_state": SEED, "n_jobs": -1, "verbosity": 0}
    best_params_per_product[product] = best_p

    with mlflow.start_run(run_name=f"xgboost_tuned_{product.replace(' ','_')}"):
        mlflow.set_tag("model_type", "xgboost_tuned")
        mlflow.set_tag("product", product)
        mlflow.log_param("n_optuna_trials", N_TRIALS)
        mlflow.log_params(best_p)
        model_tuned, _ = train(product, df_tr, best_p)
        metrics_tuned  = evaluate(model_tuned, df_te, product)
        mlflow.log_metrics({"mae": metrics_tuned["mae"], "rmse": metrics_tuned["rmse"], "mape": metrics_tuned["mape"]})
        mlflow.xgboost.log_model(model_tuned, artifact_path="model")
        tuned_models[product]  = model_tuned
        tuned_results[product] = metrics_tuned

    print(f"MAPE {xgb_results[product]['mape']:.1f}% → {metrics_tuned['mape']:.1f}%")

avg_mape_tuned = float(np.nanmean([v["mape"] for v in tuned_results.values()]))
print(f"  Tuned XGBoost avg MAPE={avg_mape_tuned:.1f}%")

# ===========================================================================
# Section 8 — Generate forecasts
# ===========================================================================
print("\n" + "=" * 60)
print("Section 8 — Generating 7-day and 30-day forecasts")

def generate_forecasts(product, model, df_history, horizon, ci_multiplier=1.645):
    history = df_history[df_history["product"] == product].sort_values("date").copy()
    revenue_buffer = dict(zip(history["date"], history["revenue"]))
    df_te   = df_test_all[df_test_all["product"] == product].copy()
    y_true  = df_te["revenue"].values
    y_pred_test = np.clip(model.predict(df_te[ALL_FEATURES]), 0, None)
    residual_std = float(np.std(y_true - y_pred_test)) if len(y_true) > 1 else float(np.mean(y_true) * 0.2)

    last_date = history["date"].max()
    records   = []
    for step in range(1, horizon + 1):
        fd = last_date + pd.Timedelta(days=step)
        def gr(d): return revenue_buffer.get(d, 0.0)
        feat_row = pd.DataFrame([{
            "lag_1": gr(fd - pd.Timedelta(days=1)),
            "lag_7": gr(fd - pd.Timedelta(days=7)),
            "lag_14": gr(fd - pd.Timedelta(days=14)),
            "rolling_mean_7":  np.mean([gr(fd - pd.Timedelta(days=d)) for d in range(1, 8)]),
            "rolling_mean_14": np.mean([gr(fd - pd.Timedelta(days=d)) for d in range(1, 15)]),
            "day_of_week": fd.dayofweek,
            "is_weekend":  int(fd.dayofweek >= 5),
            "month":       fd.month,
        }])
        point = float(np.clip(model.predict(feat_row[ALL_FEATURES])[0], 0, None))
        revenue_buffer[fd] = point
        records.append({
            "product":           product,
            "date":              fd,
            "predicted_revenue": point,
            "lower_bound":       max(0.0, point - ci_multiplier * residual_std),
            "upper_bound":       point + ci_multiplier * residual_std,
        })
    return pd.DataFrame(records)

df_full_history = df_feat.copy()
fc7_list, fc30_list = [], []
for product in FORECAST_PRODUCTS:
    model = tuned_models[product]
    fc7  = generate_forecasts(product, model, df_full_history, 7)
    fc30 = generate_forecasts(product, model, df_full_history, 30)
    fc7_list.append(fc7); fc30_list.append(fc30)
    print(f"  {product:<25} 7d={fc7['predicted_revenue'].sum():,.0f}  30d={fc30['predicted_revenue'].sum():,.0f}")

df_fc7  = pd.concat(fc7_list,  ignore_index=True); df_fc7["horizon"]  = 7
df_fc30 = pd.concat(fc30_list, ignore_index=True); df_fc30["horizon"] = 30
df_all  = pd.concat([df_fc7, df_fc30], ignore_index=True)
df_all.to_parquet(PREDICTIONS_PATH, index=False, engine="pyarrow")
print(f"\n  ✅ Forecasts saved → {PREDICTIONS_PATH}")

# ===========================================================================
# Section 9 — Register in MLflow Model Registry
# ===========================================================================
print("\n" + "=" * 60)
print("Section 9 — Registering champion in MLflow Model Registry")

from mlflow.tracking import MlflowClient
client = MlflowClient()
REGISTRY_NAME = "demand_forecast_xgboost"

try:
    client.get_registered_model(REGISTRY_NAME)
except Exception:
    client.create_registered_model(REGISTRY_NAME,
        description="XGBoost lag-feature demand forecasting champion — tuned via Optuna")

for product in FORECAST_PRODUCTS:
    with mlflow.start_run(run_name=f"xgboost_registry_{product.replace(' ','_')}") as run:
        mlflow.set_tag("model_type", "xgboost_tuned_champion")
        mlflow.set_tag("product", product)
        mlflow.log_params(best_params_per_product[product])
        mlflow.log_metrics(tuned_results[product])
        mlflow.xgboost.log_model(tuned_models[product], artifact_path="model",
                                 registered_model_name=REGISTRY_NAME)
        print(f"  Registered {product} (run {run.info.run_id[:8]}...)")

try:
    all_versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    latest = str(max(int(v.version) for v in all_versions))
    client.transition_model_version_stage(REGISTRY_NAME, latest, "Production",
                                         archive_existing_versions=True)
    print(f"\n  ✅ Version {latest} promoted to Production")
except Exception as e:
    print(f"  Note: stage transition skipped ({e})")

print("\n" + "=" * 60)
print("🏁  Sub-Tarefa 4 complete")
print(f"    Forecasts  : {PREDICTIONS_PATH}")
print(f"    MLflow     : {mlflow.get_tracking_uri()}")
print(f"    Champion   : {CHAMPION_NAME}  (tuned avg MAPE={avg_mape_tuned:.1f}%)")
