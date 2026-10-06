"""
data_loader.py — Funções de carga de dados para o dashboard.
Lê parquets e CSV pré-computados; NÃO faz inferência.
Todas as funções utilizam cache simples (carregam uma vez por processo).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Resolve o diretório raiz do projeto (coffee_sales/) a partir deste arquivo
_HERE = Path(__file__).resolve()
# src/dashboard/data_loader.py  →  coffee_sales/
_ROOT = _HERE.parent.parent.parent

RAW_CSV = _ROOT / "data" / "raw" / "coffee_sales.csv"
DEMAND_FORECAST = _ROOT / "data" / "predictions" / "demand_forecast.parquet"
CHURN_PREDICTIONS = _ROOT / "data" / "predictions" / "churn_predictions.parquet"
CUSTOMER_SEGMENTS = _ROOT / "data" / "features" / "customer_segments.parquet"
RFM_FEATURES = _ROOT / "data" / "features" / "rfm_features.parquet"
MLRUNS_DIR = _ROOT / "mlruns"


@lru_cache(maxsize=1)
def load_transactions() -> pd.DataFrame:
    """Carrega o CSV de transações e adiciona colunas de conveniência."""
    df = pd.read_csv(RAW_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Mon … 6=Sun
    df["day_name"] = df["datetime"].dt.day_name()
    return df


@lru_cache(maxsize=1)
def load_demand_forecast() -> pd.DataFrame:
    """Carrega as previsões de demanda (7 e 30 dias) por produto."""
    df = pd.read_parquet(DEMAND_FORECAST)
    df["date"] = pd.to_datetime(df["date"])
    return df


@lru_cache(maxsize=1)
def load_churn_predictions() -> pd.DataFrame:
    """Carrega as probabilidades de churn por cliente."""
    return pd.read_parquet(CHURN_PREDICTIONS)


@lru_cache(maxsize=1)
def load_customer_segments() -> pd.DataFrame:
    """Carrega os segmentos RFM por cliente."""
    return pd.read_parquet(CUSTOMER_SEGMENTS)


@lru_cache(maxsize=1)
def load_rfm_features() -> pd.DataFrame:
    """Carrega as features RFM normalizadas."""
    return pd.read_parquet(RFM_FEATURES)


@lru_cache(maxsize=1)
def load_mlflow_metrics() -> dict:
    """
    Lê métricas dos modelos diretamente do MLflow tracking store.
    Retorna um dicionário estruturado com as métricas dos experimentos.
    """
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=str(MLRUNS_DIR))

        result: dict = {
            "demand": {},
            "churn": {},
            "clustering": {},
            "monitoring": {},
            "demand_history": [],
        }

        # Experimento de forecasting
        for exp in client.search_experiments():
            runs = client.search_runs(
                exp.experiment_id,
                order_by=["start_time DESC"],
                max_results=10,
            )
            if not runs:
                continue

            if "demand" in exp.name.lower():
                best = min(runs, key=lambda r: r.data.metrics.get("mape", float("inf")))
                result["demand"] = {**best.data.metrics, "run_name": best.data.tags.get("mlflow.runName", "")}
                # Histórico de MAPE (todos os runs, ordenados por data)
                history = [
                    {
                        "run_name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                        "mape": r.data.metrics.get("mape"),
                        "start_time": r.info.start_time,
                    }
                    for r in runs
                    if "mape" in r.data.metrics
                ]
                result["demand_history"] = sorted(history, key=lambda x: x["start_time"])

            elif "churn" in exp.name.lower():
                best = max(runs, key=lambda r: r.data.metrics.get("pr_auc", 0))
                result["churn"] = {**best.data.metrics, "run_name": best.data.tags.get("mlflow.runName", "")}

            elif "cluster" in exp.name.lower():
                best = max(runs, key=lambda r: r.data.metrics.get("silhouette", 0))
                result["clustering"] = {**best.data.metrics, "run_name": best.data.tags.get("mlflow.runName", "")}

            elif "monitor" in exp.name.lower():
                result["monitoring"] = {**runs[0].data.metrics, "run_name": runs[0].data.tags.get("mlflow.runName", "")}

        return result

    except Exception as exc:  # pylint: disable=broad-except
        return {"error": str(exc), "demand": {}, "churn": {}, "clustering": {}, "monitoring": {}, "demand_history": []}
