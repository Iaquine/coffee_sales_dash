"""
retraining_pipeline.py — Automated model re-training triggered by monitoring alerts.

This pipeline:
1. Loads the latest monitoring state from MLflow and local predictions
2. Runs all alert checks via src.monitoring.alerts
3. If any alert is 'critical', triggers re-training for the affected model
4. Logs re-training results back to MLflow with tag trigger=auto_retrain

Usage
-----
    python pipelines/retraining_pipeline.py [--experiment monitoring_retrain]

The script exits with code 0 on success (re-train or no action needed),
and code 1 on unhandled errors.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root (coffee_sales/) is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
import pandas as pd

from src.monitoring.alerts import run_all_alerts

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("retraining_pipeline")

# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = str(ROOT / "mlruns")
DEFAULT_EXPERIMENT = "monitoring_retrain"


def _get_latest_run_metrics(experiment_name: str, model_name: str) -> dict:
    """Retrieve the most recent metrics for a model from MLflow."""
    try:
        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            logger.warning("MLflow experiment '%s' not found — using defaults.", experiment_name)
            return {}
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.model_name = '{model_name}'",
            order_by=["start_time DESC"],
            max_results=2,
        )
        if not runs:
            logger.warning("No MLflow runs found for model='%s'.", model_name)
            return {}
        return runs[0].data.metrics
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch MLflow metrics for '%s': %s", model_name, exc)
        return {}


def _log_retrain_result(
    experiment_name: str,
    model_name: str,
    metrics: dict,
    trigger_reason: str,
) -> str:
    """Log a re-training run to MLflow and return the run ID."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"auto_retrain_{model_name}") as run:
        mlflow.set_tag("trigger", "auto_retrain")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("trigger_reason", trigger_reason)
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value))
            except (TypeError, ValueError):
                mlflow.set_tag(key, str(value))
        run_id = run.info.run_id

    logger.info("Logged re-train run for '%s' → run_id=%s", model_name, run_id)
    return run_id


# ---------------------------------------------------------------------------
# Model re-training functions
# ---------------------------------------------------------------------------


def _retrain_demand_forecast() -> dict:
    """Re-train the demand forecasting XGBoost model."""
    logger.info("Re-training demand_forecast_xgboost …")
    try:
        from src.features.demand_features import build_demand_features
        from src.models.demand.train import train as demand_train

        df_raw = pd.read_csv(ROOT / "data" / "raw" / "coffee_sales.csv")
        df_features = build_demand_features(df_raw)

        products = df_features["coffee_name"].unique() if "coffee_name" in df_features.columns else []
        all_metrics: list[dict] = []

        for product in products:
            df_prod = df_features[df_features["coffee_name"] == product].copy()
            if len(df_prod) < 20:
                continue
            _model, metrics = demand_train(product=product, df_train=df_prod)
            all_metrics.append(metrics)

        if all_metrics:
            import numpy as np
            avg_mape = float(np.mean([m.get("mape", float("nan")) for m in all_metrics]))
            avg_mae = float(np.mean([m.get("mae", float("nan")) for m in all_metrics]))
            return {"mape": avg_mape, "mae": avg_mae, "n_products": len(all_metrics)}
        return {"status": "no_products_trained"}

    except Exception as exc:  # noqa: BLE001
        logger.error("demand forecast re-training failed: %s", exc, exc_info=True)
        return {"error": str(exc)}


def _retrain_churn() -> dict:
    """Re-train the churn classifier model."""
    logger.info("Re-training churn_classifier …")
    try:
        from src.features.churn_features import build_churn_features
        from src.models.churn.train import train as churn_train
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import average_precision_score

        df_rfm = pd.read_parquet(ROOT / "data" / "features" / "rfm_features.parquet")
        df_features = build_churn_features(df_rfm)

        feature_cols = [c for c in df_features.columns if c not in ("card_id", "churned")]
        X = df_features[feature_cols].values
        y = df_features["churned"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = churn_train(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        pr_auc = float(average_precision_score(y_test, y_proba))

        return {"pr_auc": pr_auc, "n_train": len(X_train), "n_test": len(X_test)}

    except Exception as exc:  # noqa: BLE001
        logger.error("churn re-training failed: %s", exc, exc_info=True)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Monitoring report builder (reads from local artefacts / MLflow)
# ---------------------------------------------------------------------------


def _build_monitoring_report() -> dict:
    """Build a minimal monitoring report from available artefacts.

    Falls back to synthetic sentinel values when artefacts are missing so the
    pipeline can still run end-to-end in CI / demo environments.
    """
    report: dict = {}

    # --- Demand forecasting ---
    try:
        forecast_df = pd.read_parquet(ROOT / "data" / "predictions" / "demand_forecast.parquet")
        if "mape" in forecast_df.columns:
            mapes = forecast_df["mape"].dropna().values
            if len(mapes) >= 2:
                half = len(mapes) // 2
                baseline_mape = float(mapes[:half].mean())
                current_mape = float(mapes[half:].mean())
            else:
                baseline_mape = float(mapes[0])
                current_mape = float(mapes[0])
            report["forecasting"] = {
                "baseline_mape": baseline_mape,
                "current_mape": current_mape,
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load forecast predictions: %s", exc)
        report["forecasting"] = {"baseline_mape": 0.20, "current_mape": 0.20}

    # --- Churn ---
    try:
        churn_df = pd.read_parquet(ROOT / "data" / "predictions" / "churn_predictions.parquet")
        # Use churn score variance as a proxy for drift in the absence of live labels
        if "churn_probability" in churn_df.columns:
            scores = churn_df["churn_probability"].dropna().values
            half = len(scores) // 2
            baseline_pr_auc = 0.75  # typical value from training run
            current_pr_auc = 0.75 - (scores[half:].mean() - scores[:half].mean()) * 0.5
            report["churn"] = {
                "baseline_pr_auc": max(0.0, baseline_pr_auc),
                "current_pr_auc": max(0.0, float(current_pr_auc)),
            }
        else:
            report["churn"] = {"baseline_pr_auc": 0.75, "current_pr_auc": 0.75}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load churn predictions: %s", exc)
        report["churn"] = {"baseline_pr_auc": 0.75, "current_pr_auc": 0.75}

    # --- Data drift (PSI placeholder — run drift_detector separately) ---
    report["drift"] = {"psi_value": 0.05}  # default: stable

    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_retraining_pipeline(experiment_name: str = DEFAULT_EXPERIMENT) -> None:
    """Execute the full re-training pipeline."""
    logger.info("=== Retraining Pipeline Start ===")

    monitoring_report = _build_monitoring_report()
    alerts = run_all_alerts(monitoring_report)

    critical_alerts = [a for a in alerts if a["severity"] == "critical"]
    logger.info(
        "Alerts: %d total, %d critical — %s",
        len(alerts),
        len(critical_alerts),
        [a["model"] for a in critical_alerts] if critical_alerts else "none",
    )

    if not critical_alerts:
        logger.info("No critical alerts — re-training not required. Pipeline finished.")
        return

    retrained_models: list[str] = []

    for alert in critical_alerts:
        model = alert["model"]
        reason = alert["message"]

        if "demand_forecast" in model:
            metrics = _retrain_demand_forecast()
            run_id = _log_retrain_result(experiment_name, model, metrics, reason)
            retrained_models.append(f"{model} (run_id={run_id})")

        elif "churn" in model:
            metrics = _retrain_churn()
            run_id = _log_retrain_result(experiment_name, model, metrics, reason)
            retrained_models.append(f"{model} (run_id={run_id})")

        elif "drift" in model:
            # Data drift triggers re-training of all models
            for retrain_fn, retrain_model in [
                (_retrain_demand_forecast, "demand_forecast_xgboost"),
                (_retrain_churn, "churn_classifier"),
            ]:
                metrics = retrain_fn()
                run_id = _log_retrain_result(experiment_name, retrain_model, metrics, reason)
                retrained_models.append(f"{retrain_model} (run_id={run_id})")

    logger.info("=== Retraining Pipeline Complete — models retrained: %s ===", retrained_models)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated re-training pipeline")
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="MLflow experiment name for re-training runs",
    )
    args = parser.parse_args()

    try:
        run_retraining_pipeline(experiment_name=args.experiment)
    except Exception as exc:
        logger.error("Pipeline failed with unhandled error: %s", exc, exc_info=True)
        sys.exit(1)
