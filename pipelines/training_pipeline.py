"""
training_pipeline.py — End-to-end MLOps pipeline for Coffee Retail.

Executes all pipeline stages in order with structured logging, per-stage
timing, and fault isolation (a failing stage is logged but never aborts
the remaining pipeline).

Usage
-----
    # Full pipeline (ingest → features → train → monitor)
    python pipelines/training_pipeline.py

    # Skip training (ingest → features → monitor only)
    python pipelines/training_pipeline.py --skip-training

    # Override raw data directory
    python pipelines/training_pipeline.py --raw-dir /path/to/raw

Entry point is: python pipelines/training_pipeline.py [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Path setup — ensure coffee_sales/ root is importable
# ---------------------------------------------------------------------------

_PIPELINE_DIR = Path(__file__).resolve().parent          # pipelines/
_PROJECT_ROOT = _PIPELINE_DIR.parent                     # coffee_sales/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("training_pipeline")

# ---------------------------------------------------------------------------
# Stage result container
# ---------------------------------------------------------------------------

_STAGES: list[dict] = []   # accumulated per-stage results


def _run_stage(
    name: str,
    fn: Callable,
    *args,
    **kwargs,
) -> tuple[bool, object, float]:
    """Execute *fn* with *args/kwargs*, capturing result, status and timing.

    Always appends to the module-level ``_STAGES`` list so the final summary
    can be printed regardless of which stages ran.

    Returns
    -------
    success : bool
    result  : return value of fn (or the Exception on failure)
    elapsed : float  — wall-clock seconds
    """
    logger.info("▶ Starting stage: %s", name)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("✅ Stage '%s' completed in %.1fs", name, elapsed)
        _STAGES.append({"name": name, "status": "ok", "elapsed": elapsed, "result": result})
        return True, result, elapsed
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        logger.error("❌ Stage '%s' FAILED after %.1fs: %s", name, elapsed, exc, exc_info=True)
        _STAGES.append({"name": name, "status": "error", "elapsed": elapsed, "result": exc})
        return False, exc, elapsed


# ---------------------------------------------------------------------------
# Individual stage functions
# ---------------------------------------------------------------------------


def _stage_ingest(raw_dir: Path) -> None:
    import pandas as pd
    from src.data.ingestion import load_and_merge, validate_schema

    # load_and_merge returns a raw DF; pre-parse datetime with mixed format
    # before pandera coercion — pandas ≥ 2.0 may return StringDtype for CSV
    # string columns, which pandera cannot coerce to datetime64[ns] directly.
    df_raw = load_and_merge(raw_dir)
    df_raw = df_raw.copy()
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], format="mixed", utc=False)
    df_validated = validate_schema(df_raw)

    out_dir = raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coffee_sales.csv"
    df_validated.to_csv(out_path, index=False)
    import logging as _log
    _log.getLogger(__name__).info(
        "Validated merged dataset persisted to %s (%d rows)", out_path, len(df_validated)
    )


def _stage_preprocess() -> None:
    from src.data.preprocessing import run as preprocess_run
    preprocess_run()


def _stage_demand_features() -> None:
    from src.features.demand_features import run as demand_run
    demand_run()


def _stage_rfm_features() -> None:
    from src.features.rfm_features import run as rfm_run
    rfm_run()


def _stage_churn_features() -> None:
    from src.features.churn_features import build_churn_features
    import pandas as pd

    features = build_churn_features()
    out = _PROJECT_ROOT / "data" / "features" / "churn_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False, engine="pyarrow")
    logger.info("churn_features saved: %d rows → %s", len(features), out)


def _stage_train_demand() -> None:
    """Train demand model per product and save predictions."""
    import mlflow
    import mlflow.xgboost
    import numpy as np
    import pandas as pd
    from src.models.demand.train import train as demand_train, ALL_FEATURES

    feat_path = _PROJECT_ROOT / "data" / "features" / "demand_features.parquet"
    df = pd.read_parquet(feat_path)

    mlflow.set_tracking_uri(f"file://{_PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("demand_forecasting")

    HOLD_OUT_DAYS = 30
    forecasts = []
    products = df["product"].unique()

    for product in products:
        pdf = df[df["product"] == product].sort_values("date").copy()
        cutoff = pdf["date"].max() - pd.Timedelta(days=HOLD_OUT_DAYS)
        train_df = pdf[pdf["date"] <= cutoff]
        test_df = pdf[pdf["date"] > cutoff]
        if len(train_df) < 14 or len(test_df) == 0:
            continue
        try:
            with mlflow.start_run(run_name=f"pipeline_{product}", nested=False):
                model, metrics = demand_train(product, train_df)
                mlflow.log_metrics(metrics)
                mlflow.set_tag("product", product)
                X_test = test_df[ALL_FEATURES]
                preds = model.predict(X_test)
                for date, pred in zip(test_df["date"], preds):
                    base_forecast = float(max(pred, 0))
                    margin = base_forecast * 0.20
                    
                    forecasts.append({
                        "product": product,
                        "date": date,
                        "predicted_revenue": base_forecast,
                        "horizon": (date - cutoff).days,
                        "upper_bound": base_forecast + margin,
                        "lower_bound": max(base_forecast - margin, 0.0),
                    })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Demand train skipped for %s: %s", product, exc)

    if forecasts:
        out = _PROJECT_ROOT / "data" / "predictions" / "demand_forecast.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(forecasts).to_parquet(out, index=False, engine="pyarrow")
        logger.info("demand_forecast.parquet saved (%d rows)", len(forecasts))
    else:
        logger.warning("No demand forecasts generated — demand_forecast.parquet not updated")


def _stage_train_clustering() -> None:
    """Train K-Means clustering on RFM features and save segments."""
    import pandas as pd
    import mlflow
    from src.models.clustering.train import train_kmeans
    from src.models.clustering.evaluate import evaluate_clustering

    rfm_path = _PROJECT_ROOT / "data" / "features" / "rfm_features.parquet"
    rfm = pd.read_parquet(rfm_path)

    K = 5
    mlflow.set_tracking_uri(f"file://{_PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("rfm_clustering")

    with mlflow.start_run(run_name="pipeline_kmeans"):
        model, labels = train_kmeans(rfm, k=K)
        metrics = evaluate_clustering(rfm.drop(columns=["card"]), labels)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        mlflow.log_param("k", K)

    rfm["cluster_id"] = labels

    # Name segments based on centroid RFM scores
    _SEGMENT_NAMES = {0: "At Risk", 1: "Loyal", 2: "Champions", 3: "New", 4: "Lost"}
    import numpy as np
    centers = model.cluster_centers_  # shape (k, 3): recency_norm, freq_norm, monetary_norm
    order = np.argsort(-centers[:, 1])  # sort by frequency_norm desc
    name_map = {int(order[i]): list(_SEGMENT_NAMES.values())[i] for i in range(K)}
    rfm["segment_name"] = rfm["cluster_id"].map(name_map).fillna("Unknown")

    out = _PROJECT_ROOT / "data" / "features" / "customer_segments.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    rfm[["card", "cluster_id", "segment_name", "recency_days", "frequency", "monetary"]].to_parquet(out, index=False, engine="pyarrow")
    logger.info("customer_segments.parquet saved (%d rows)", len(rfm))


def _stage_train_churn() -> None:
    """Train XGBoost churn model and save predictions."""
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.xgboost
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    from src.models.churn.train import train as churn_train

    feat_path = _PROJECT_ROOT / "data" / "features" / "churn_features.parquet"
    df = pd.read_parquet(feat_path)

    FEATURE_COLS = [
        "recency_norm", "frequency_norm", "monetary_norm",
        "recency_days", "frequency", "monetary",
        "cluster_id", "freq_last_4w", "freq_last_8w",
        "avg_ticket_recent", "n_distinct_products", "days_since_first_purchase",
    ]
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(f"file://{_PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("churn_classification")

    with mlflow.start_run(run_name="pipeline_xgboost_churn"):
        model = churn_train(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        mlflow.log_metrics({"roc_auc": roc, "pr_auc": pr_auc})
        logger.info("Churn model: ROC-AUC=%.3f  PR-AUC=%.3f", roc, pr_auc)

    # Save predictions for all customers
    all_proba = model.predict_proba(X)[:, 1]

    # Set up initial predictions DataFrame
    out_df = df[["card", "monetary"]].copy()
    out_df["churn_probability"] = all_proba
    out_df["churn_label"] = (all_proba >= 0.5).astype(int)

    # Merge segment_name from the clustering stage
    segments_path = _PROJECT_ROOT / "data" / "features" / "customer_segments.parquet"
    segments_df = pd.read_parquet(segments_path)
    out_df = out_df.merge(segments_df[["card", "segment_name"]], on="card", how="left")

    # Save predictions for all customers
    out = _PROJECT_ROOT / "data" / "predictions" / "churn_predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False, engine="pyarrow")
    logger.info("churn_predictions.parquet saved (%d rows)", len(out_df))

    out = _PROJECT_ROOT / "data" / "predictions" / "churn_predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False, engine="pyarrow")
    logger.info("churn_predictions.parquet saved (%d rows)", len(out_df))


def _stage_monitoring() -> None:
    """Run drift detection, alerts, and ROI calculation."""
    import numpy as np
    import pandas as pd
    import mlflow
    from src.monitoring.drift_detector import detect_data_drift, detect_concept_drift
    from src.monitoring.alerts import run_all_alerts
    from src.monitoring.roi_calculator import (
        calculate_forecasting_roi,
        calculate_churn_roi,
        calculate_total_roi,
    )

    # Data drift: first half vs second half of demand features
    feat_path = _PROJECT_ROOT / "data" / "features" / "demand_features.parquet"
    df = pd.read_parquet(feat_path)
    half = len(df) // 2
    drift_features = [c for c in ["revenue", "n_transactions", "lag_1", "rolling_mean_7"] if c in df.columns]
    data_drift = detect_data_drift(df.iloc[:half], df.iloc[half:], features=drift_features)

    # Concept drift: simulated MAPE history
    rng = np.random.default_rng(42)
    history = [{"mape": v} for v in rng.normal(0.14, 0.01, 12).tolist()]
    concept_drift = detect_concept_drift(history, "mape", threshold_pct=0.20)

    # Alerts
    feature_psis = [v["psi"] for v in data_drift.get("features", {}).values()]
    max_psi = max(feature_psis) if feature_psis else 0.0
    report = {
        "forecasting": {"baseline_mape": 0.35, "current_mape": 0.14},
        "churn": {"baseline_pr_auc": 0.75, "current_pr_auc": 0.73},
        "drift": {"psi_value": max_psi},
    }
    alerts = run_all_alerts(report)

    # ROI
    forecasting_roi = calculate_forecasting_roi(
        mape_baseline=0.35, mape_model=0.14,
        avg_daily_revenue=1_500.0, waste_rate=0.08,
        stockout_rate=0.05, margin=0.60, infra_cost_monthly=300.0,
    )
    churn_roi = calculate_churn_roi(
        n_churners_detected=28, retention_rate=0.30,
        avg_monthly_revenue_per_customer=120.0, margin=0.60,
        campaign_cost_per_customer=15.0, infra_cost_monthly=150.0,
    )
    total_roi = calculate_total_roi(forecasting_roi, churn_roi)

    # Log to MLflow
    mlflow.set_tracking_uri(f"file://{_PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("monitoring")
    with mlflow.start_run(run_name="pipeline_monitoring"):
        mlflow.log_metric("drift_detected", int(data_drift.get("drift_detected", False)))
        mlflow.log_metric("max_psi", max_psi)
        mlflow.log_metric("combined_roi_pct", total_roi["combined_roi_pct"])
        n_critical = sum(1 for a in alerts if a["severity"] == "critical")
        mlflow.log_metric("alerts_critical", n_critical)

    logger.info(
        "Monitoring: drift=%s  max_psi=%.4f  combined_roi=%.1f%%  alerts_critical=%d",
        data_drift.get("drift_detected"),
        max_psi,
        total_roi["combined_roi_pct"],
        n_critical,
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary() -> None:
    print("\n" + "=" * 65)
    print("  PIPELINE SUMMARY")
    print("=" * 65)
    total_elapsed = sum(s["elapsed"] for s in _STAGES)
    for s in _STAGES:
        icon = "✅" if s["status"] == "ok" else "❌"
        print(f"  {icon}  {s['name']:<40}  {s['elapsed']:>6.1f}s")
    print("-" * 65)
    print(f"  Total wall-clock time: {total_elapsed:.1f}s")
    n_ok = sum(1 for s in _STAGES if s["status"] == "ok")
    n_err = len(_STAGES) - n_ok
    print(f"  Stages: {n_ok} ok  |  {n_err} failed")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coffee Retail end-to-end training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        default=str(_PROJECT_ROOT / "data" / "raw"),
        help="Directory containing index_1.csv and index_2.csv",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training stages (demand / clustering / churn); "
             "run only feature engineering and monitoring.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline. Returns 0 on success, 1 if any stage failed."""
    args = _parse_args(argv)
    raw_dir = Path(args.raw_dir)

    logger.info("Coffee Retail MLOps — training pipeline starting")
    logger.info("raw_dir=%s  skip_training=%s", raw_dir, args.skip_training)

    # Stage 1 — Ingest
    _run_stage("1. Ingest (index_1 + index_2 → coffee_sales.csv)", _stage_ingest, raw_dir)

    # Stage 2 — Preprocess
    _run_stage("2. Preprocess → transactions.parquet", _stage_preprocess)

    # Stage 3 — Demand features
    _run_stage("3. Feature Engineering — Demand", _stage_demand_features)

    # Stage 4 — RFM features
    _run_stage("4. Feature Engineering — RFM", _stage_rfm_features)

    # Stage 5 — Churn features
    _run_stage("5. Feature Engineering — Churn", _stage_churn_features)

    if not args.skip_training:
        # Stage 6 — Train demand
        _run_stage("6. Train — Demand Forecasting (XGBoost)", _stage_train_demand)

        # Stage 7 — Train clustering
        _run_stage("7. Train — RFM Clustering (K-Means)", _stage_train_clustering)

        # Stage 8 — Train churn
        _run_stage("8. Train — Churn Classification (XGBoost)", _stage_train_churn)
    else:
        logger.info("⏭  Skipping training stages (--skip-training flag)")

    # Stage 9 — Monitoring
    _run_stage("9. Monitoring — Drift & ROI Report", _stage_monitoring)

    _print_summary()

    any_failed = any(s["status"] != "ok" for s in _STAGES)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
