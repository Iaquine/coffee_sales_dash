"""
run_monitoring.py — End-to-end MLOps monitoring demonstration.

This script:
  1. Detects data drift (first-half vs second-half of the raw dataset)
  2. Detects concept drift from a synthetic metrics history
  3. Runs all model health alerts
  4. Calculates ROI with documented example inputs (see FINANCIAL PARAMETERS below)
  5. Logs all metrics to MLflow under the 'monitoring' experiment
  6. Prints a full text report to stdout

Usage
-----
    cd coffee_sales/
    python run_monitoring.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
import numpy as np
import pandas as pd

from src.monitoring.alerts import run_all_alerts
from src.monitoring.drift_detector import detect_concept_drift, detect_data_drift
from src.monitoring.roi_calculator import (
    calculate_churn_roi,
    calculate_forecasting_roi,
    calculate_total_roi,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_monitoring")

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = str(ROOT / "mlruns")
EXPERIMENT_NAME = "monitoring"

# ---------------------------------------------------------------------------
# FINANCIAL PARAMETERS (documented here — zero hardcode in roi_calculator.py)
# ---------------------------------------------------------------------------
#
# These values are representative inputs for a mid-size coffee retail chain.
# In production they are provided by the user via the Dash dashboard (Aba 4).
#
# avg_daily_revenue_brl    : R$ 1 500 / day (based on ~3 898 txns / 365 days × R$38.7 avg ticket)
# waste_rate               : 8 % — estimated spoilage rate without ML-guided stock levels
# stockout_rate            : 5 % — estimated lost sales from stockout events
# margin                   : 60 % — gross margin on coffee products
# infra_cost_monthly_brl   : R$ 300 / month — cloud VM + MLflow hosting
# mape_baseline            : 35 % — naive last-week-same-day baseline
# mape_model               : 14 % — XGBoost champion (approximated from training logs)
#
# n_churners_detected      : 28 — high-risk customers flagged this month
# retention_rate           : 30 % — fraction successfully retained after outreach campaign
# avg_monthly_rev_per_cust : R$ 120 / month per retained customer
# campaign_cost_per_cust   : R$ 15 — per-customer campaign cost (email / SMS outreach)

FINANCIAL_PARAMS = {
    # Forecasting
    "mape_baseline": 0.35,
    "mape_model": 0.14,
    "avg_daily_revenue": 1_500.0,
    "waste_rate": 0.08,
    "stockout_rate": 0.05,
    "margin": 0.60,
    "infra_cost_monthly_forecast": 300.0,
    # Churn
    "n_churners_detected": 28,
    "retention_rate": 0.30,
    "avg_monthly_revenue_per_customer": 120.0,
    "campaign_cost_per_customer": 15.0,
    "infra_cost_monthly_churn": 150.0,
}

# ---------------------------------------------------------------------------
# Drift detection features
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = ["revenue", "n_transactions", "lag_1", "rolling_mean_7"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_raw() -> pd.DataFrame:
    """Load raw coffee sales CSV."""
    path = ROOT / "data" / "raw" / "coffee_sales.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _load_demand_features() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "features" / "demand_features.parquet")


def _load_churn_predictions() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "predictions" / "churn_predictions.parquet")


def _print_section(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_dict(d: dict, indent: int = 4) -> None:
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _print_dict(v, indent + 4)
        elif isinstance(v, float):
            print(f"{pad}{k}: {v:.4f}")
        else:
            print(f"{pad}{k}: {v}")


# ---------------------------------------------------------------------------
# Step 1 — Data drift
# ---------------------------------------------------------------------------


def run_data_drift_detection() -> dict:
    """Compare first-half vs second-half of the demand features dataset."""
    logger.info("Running data drift detection …")
    df = _load_demand_features()
    half = len(df) // 2
    df_reference = df.iloc[:half].copy()
    df_current = df.iloc[half:].copy()

    available_features = [f for f in NUMERIC_FEATURES if f in df.columns]
    result = detect_data_drift(df_reference, df_current, features=available_features)
    return result


# ---------------------------------------------------------------------------
# Step 2 — Concept drift
# ---------------------------------------------------------------------------


def run_concept_drift_detection() -> dict:
    """Simulate a MAPE history and check for concept drift."""
    logger.info("Running concept drift detection …")

    # Simulate 12 months of MAPE observations: stable for 6 months then gradual increase
    rng = np.random.default_rng(42)
    stable = rng.normal(loc=0.14, scale=0.01, size=6).tolist()
    degraded = rng.normal(loc=0.18, scale=0.015, size=6).tolist()
    metrics_history = [{"mape": v, "month": i + 1} for i, v in enumerate(stable + degraded)]

    result = detect_concept_drift(
        metrics_history=metrics_history,
        metric_key="mape",
        threshold_pct=0.20,
    )
    return result


# ---------------------------------------------------------------------------
# Step 3 — Alerts
# ---------------------------------------------------------------------------


def run_alert_checks(data_drift_result: dict) -> list[dict]:
    """Build a monitoring report and evaluate all alerts."""
    logger.info("Running alert checks …")

    # Derive max PSI from data drift
    feature_psis = [v["psi"] for v in data_drift_result.get("features", {}).values()]
    max_psi = max(feature_psis) if feature_psis else 0.0

    monitoring_report = {
        "forecasting": {
            "baseline_mape": FINANCIAL_PARAMS["mape_baseline"],
            "current_mape": FINANCIAL_PARAMS["mape_model"],
            "threshold_pct": 0.20,
        },
        "churn": {
            "baseline_pr_auc": 0.75,
            "current_pr_auc": 0.73,
            "threshold_pct": 0.10,
        },
        "drift": {
            "psi_value": max_psi,
        },
    }

    alerts = run_all_alerts(monitoring_report)
    return alerts


# ---------------------------------------------------------------------------
# Step 4 — ROI calculation
# ---------------------------------------------------------------------------


def run_roi_calculation() -> dict:
    """Calculate forecasting and churn ROI using the documented parameters."""
    logger.info("Calculating ROI …")
    p = FINANCIAL_PARAMS

    forecasting_roi = calculate_forecasting_roi(
        mape_baseline=p["mape_baseline"],
        mape_model=p["mape_model"],
        avg_daily_revenue=p["avg_daily_revenue"],
        waste_rate=p["waste_rate"],
        stockout_rate=p["stockout_rate"],
        margin=p["margin"],
        infra_cost_monthly=p["infra_cost_monthly_forecast"],
    )

    churn_roi = calculate_churn_roi(
        n_churners_detected=p["n_churners_detected"],
        retention_rate=p["retention_rate"],
        avg_monthly_revenue_per_customer=p["avg_monthly_revenue_per_customer"],
        margin=p["margin"],
        campaign_cost_per_customer=p["campaign_cost_per_customer"],
        infra_cost_monthly=p["infra_cost_monthly_churn"],
    )

    total_roi = calculate_total_roi(forecasting_roi, churn_roi)

    return {
        "forecasting_roi": forecasting_roi,
        "churn_roi": churn_roi,
        "total_roi": total_roi,
    }


# ---------------------------------------------------------------------------
# Step 5 — Log to MLflow
# ---------------------------------------------------------------------------


def log_to_mlflow(
    data_drift: dict,
    concept_drift: dict,
    roi: dict,
    alerts: list[dict],
) -> str:
    """Log all monitoring metrics to MLflow under the 'monitoring' experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="monitoring_report") as run:
        # Data drift
        for feat, vals in data_drift.get("features", {}).items():
            mlflow.log_metric(f"psi_{feat}", vals["psi"])
            mlflow.log_metric(f"ks_stat_{feat}", vals["ks_statistic"])
            mlflow.log_metric(f"ks_pval_{feat}", vals["ks_p_value"])
        mlflow.log_metric("drift_detected", int(data_drift.get("drift_detected", False)))

        # Concept drift
        if concept_drift.get("baseline_value") is not None:
            mlflow.log_metric("concept_drift_baseline_mape", concept_drift["baseline_value"])
            mlflow.log_metric("concept_drift_current_mape", concept_drift["current_value"])
            mlflow.log_metric("concept_drift_degradation_pct", concept_drift["degradation_pct"])
        mlflow.log_metric("concept_drift_detected", int(concept_drift.get("drift_detected", False)))

        # ROI — forecasting
        f_roi = roi["forecasting_roi"]
        mlflow.log_metric("forecast_waste_savings", f_roi["waste_savings"])
        mlflow.log_metric("forecast_stockout_recovery", f_roi["stockout_recovery"])
        mlflow.log_metric("forecast_total_benefit", f_roi["total_benefit"])
        mlflow.log_metric("forecast_total_cost", f_roi["total_cost"])
        mlflow.log_metric("forecast_roi_pct", f_roi["roi_pct"])

        # ROI — churn
        c_roi = roi["churn_roi"]
        mlflow.log_metric("churn_revenue_saved", c_roi["revenue_saved"])
        mlflow.log_metric("churn_campaign_cost", c_roi["campaign_cost"])
        mlflow.log_metric("churn_net_benefit", c_roi["net_benefit"])
        mlflow.log_metric("churn_roi_pct", c_roi["roi_pct"])

        # ROI — total
        t_roi = roi["total_roi"]
        mlflow.log_metric("total_net_benefit", t_roi["total_net_benefit"])
        mlflow.log_metric("total_cost", t_roi["total_cost"])
        mlflow.log_metric("combined_roi_pct", t_roi["combined_roi_pct"])

        # Alerts summary
        n_critical = sum(1 for a in alerts if a["severity"] == "critical")
        n_warning = sum(1 for a in alerts if a["severity"] == "warning")
        mlflow.log_metric("alerts_critical", n_critical)
        mlflow.log_metric("alerts_warning", n_warning)

        mlflow.set_tag("pipeline", "monitoring")
        run_id = run.info.run_id

    logger.info("MLflow run logged → experiment='%s', run_id=%s", EXPERIMENT_NAME, run_id)
    return run_id


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(
    data_drift: dict,
    concept_drift: dict,
    alerts: list[dict],
    roi: dict,
    mlflow_run_id: str,
) -> None:
    """Print a full monitoring report to stdout."""
    _print_section("DATA DRIFT REPORT (first-half vs second-half of demand features)")
    print(f"  drift_detected : {data_drift['drift_detected']}")
    for feat, vals in data_drift.get("features", {}).items():
        status = "⚠ CRITICAL" if vals["psi"] > 0.2 else ("⚡ warning" if vals["psi"] > 0.1 else "✓ ok")
        print(
            f"  {feat:<25} PSI={vals['psi']:.4f} {status}  |  "
            f"KS stat={vals['ks_statistic']:.4f}  p={vals['ks_p_value']:.4f}"
        )

    _print_section("CONCEPT DRIFT REPORT (MAPE over 12 simulated months)")
    print(f"  drift_detected  : {concept_drift['drift_detected']}")
    print(f"  baseline_mape   : {concept_drift['baseline_value']:.4f}")
    print(f"  current_mape    : {concept_drift['current_value']:.4f}")
    print(f"  degradation_pct : {concept_drift['degradation_pct'] * 100:.1f}%")

    _print_section("ALERT STATUS")
    for a in alerts:
        icon = {"ok": "✓", "warning": "⚡", "critical": "⚠"}.get(a["severity"], "?")
        print(f"  [{a['severity'].upper():8s}] {icon} {a['model']}")
        print(f"             {a['message']}")

    _print_section("ROI CALCULATION")

    f = roi["forecasting_roi"]
    print("  Demand Forecasting Model")
    print(f"    waste_savings       : R$ {f['waste_savings']:>10,.2f} / month")
    print(f"    stockout_recovery   : R$ {f['stockout_recovery']:>10,.2f} / month")
    print(f"    total_benefit       : R$ {f['total_benefit']:>10,.2f} / month")
    print(f"    total_cost          : R$ {f['total_cost']:>10,.2f} / month")
    print(f"    roi_pct             :    {f['roi_pct']:>10.1f} %")
    payback = f"    payback_days        :    {f['payback_days']:>10.1f} days" if f["payback_days"] else "    payback_days        :    N/A"
    print(payback)

    c = roi["churn_roi"]
    print("\n  Churn Prevention Model")
    print(f"    customers_retained  :    {c['customers_retained']:>10.1f}")
    print(f"    revenue_saved       : R$ {c['revenue_saved']:>10,.2f} / month")
    print(f"    campaign_cost       : R$ {c['campaign_cost']:>10,.2f} / month")
    print(f"    net_benefit         : R$ {c['net_benefit']:>10,.2f} / month")
    print(f"    roi_pct             :    {c['roi_pct']:>10.1f} %")

    t = roi["total_roi"]
    print("\n  ── Combined Project ROI ──")
    print(f"    total_net_benefit   : R$ {t['total_net_benefit']:>10,.2f} / month")
    print(f"    total_cost          : R$ {t['total_cost']:>10,.2f} / month")
    print(f"    combined_roi_pct    :    {t['combined_roi_pct']:>10.1f} %")

    _print_section("MLFLOW")
    print(f"  Experiment : {EXPERIMENT_NAME}")
    print(f"  Run ID     : {mlflow_run_id}")
    print(f"  Tracking   : {MLFLOW_TRACKING_URI}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logger.info("Starting MLOps monitoring demonstration …")

    # 1. Data drift
    data_drift = run_data_drift_detection()

    # 2. Concept drift
    concept_drift = run_concept_drift_detection()

    # 3. Alerts
    alerts = run_alert_checks(data_drift)

    # 4. ROI
    roi = run_roi_calculation()

    # 5. Log to MLflow
    run_id = log_to_mlflow(data_drift, concept_drift, roi, alerts)

    # 6. Print report
    print_report(data_drift, concept_drift, alerts, roi, run_id)

    logger.info("Monitoring demonstration complete.")
