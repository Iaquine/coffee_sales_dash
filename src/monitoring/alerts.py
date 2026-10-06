"""
alerts.py — Model health alert system.

Severity levels
---------------
'ok'       — metric within acceptable bounds
'warning'  — metric approaching threshold; monitor closely
'critical' — threshold exceeded; re-training recommended

Public API
----------
check_forecasting_alert(current_mape, baseline_mape, threshold_pct=0.20) -> dict
check_churn_alert(current_pr_auc, baseline_pr_auc, threshold_pct=0.10) -> dict
check_drift_alert(psi_value) -> dict
run_all_alerts(monitoring_report) -> list[dict]
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PSI severity thresholds (per plan spec)
# ---------------------------------------------------------------------------
PSI_WARNING = 0.1
PSI_CRITICAL = 0.2


# ---------------------------------------------------------------------------
# Individual alert checks
# ---------------------------------------------------------------------------


def check_forecasting_alert(
    current_mape: float,
    baseline_mape: float,
    threshold_pct: float = 0.20,
) -> dict:
    """Alert when MAPE has degraded relative to the baseline.

    Parameters
    ----------
    current_mape : float
        MAPE of the model in the current evaluation window.
    baseline_mape : float
        MAPE recorded at model deployment / last re-train.
    threshold_pct : float
        Relative MAPE increase above which a critical alert is raised.

    Returns
    -------
    dict with keys 'model', 'alert' (bool), 'message' (str), 'severity' (str).
    """
    if baseline_mape <= 0:
        return {
            "model": "demand_forecast_xgboost",
            "alert": False,
            "message": "Baseline MAPE is zero or negative — cannot compute degradation.",
            "severity": "ok",
        }

    degradation = (current_mape - baseline_mape) / baseline_mape

    if degradation > threshold_pct:
        severity = "critical"
        alert = True
        message = (
            f"MAPE degraded {degradation * 100:.1f}% above baseline "
            f"(baseline={baseline_mape:.4f}, current={current_mape:.4f}). "
            "Re-training recommended."
        )
    elif degradation > threshold_pct * 0.5:
        severity = "warning"
        alert = True
        message = (
            f"MAPE showing early degradation {degradation * 100:.1f}% above baseline "
            f"(baseline={baseline_mape:.4f}, current={current_mape:.4f}). "
            "Monitor closely."
        )
    else:
        severity = "ok"
        alert = False
        message = (
            f"MAPE within acceptable bounds (baseline={baseline_mape:.4f}, "
            f"current={current_mape:.4f}, change={degradation * 100:+.1f}%)."
        )

    _log_alert("demand_forecast_xgboost", severity, message)
    return {
        "model": "demand_forecast_xgboost",
        "alert": alert,
        "message": message,
        "severity": severity,
    }


def check_churn_alert(
    current_pr_auc: float,
    baseline_pr_auc: float,
    threshold_pct: float = 0.10,
) -> dict:
    """Alert when PR-AUC has dropped relative to the baseline.

    Parameters
    ----------
    current_pr_auc : float
        PR-AUC of the churn model in the current evaluation window.
    baseline_pr_auc : float
        PR-AUC at deployment / last re-train.
    threshold_pct : float
        Relative PR-AUC drop below which a critical alert is raised.

    Returns
    -------
    dict with keys 'model', 'alert' (bool), 'message' (str), 'severity' (str).
    """
    if baseline_pr_auc <= 0:
        return {
            "model": "churn_classifier",
            "alert": False,
            "message": "Baseline PR-AUC is zero or negative — cannot compute degradation.",
            "severity": "ok",
        }

    # PR-AUC: lower is worse, so degradation = drop
    degradation = (baseline_pr_auc - current_pr_auc) / baseline_pr_auc

    if degradation > threshold_pct:
        severity = "critical"
        alert = True
        message = (
            f"PR-AUC dropped {degradation * 100:.1f}% below baseline "
            f"(baseline={baseline_pr_auc:.4f}, current={current_pr_auc:.4f}). "
            "Re-training recommended."
        )
    elif degradation > threshold_pct * 0.5:
        severity = "warning"
        alert = True
        message = (
            f"PR-AUC showing early decline {degradation * 100:.1f}% below baseline "
            f"(baseline={baseline_pr_auc:.4f}, current={current_pr_auc:.4f}). "
            "Monitor closely."
        )
    else:
        severity = "ok"
        alert = False
        message = (
            f"PR-AUC within acceptable bounds (baseline={baseline_pr_auc:.4f}, "
            f"current={current_pr_auc:.4f}, change={-degradation * 100:+.1f}%)."
        )

    _log_alert("churn_classifier", severity, message)
    return {
        "model": "churn_classifier",
        "alert": alert,
        "message": message,
        "severity": severity,
    }


def check_drift_alert(psi_value: float) -> dict:
    """Alert based on PSI value thresholds (per plan spec).

    PSI < 0.1  → ok
    0.1 ≤ PSI < 0.2 → warning
    PSI ≥ 0.2  → critical

    Parameters
    ----------
    psi_value : float
        Aggregated or per-feature PSI value.

    Returns
    -------
    dict with keys 'model', 'alert' (bool), 'message' (str), 'severity' (str).
    """
    if psi_value >= PSI_CRITICAL:
        severity = "critical"
        alert = True
        message = (
            f"PSI={psi_value:.4f} — significant population shift detected. "
            "Data re-collection and model re-training required."
        )
    elif psi_value >= PSI_WARNING:
        severity = "warning"
        alert = True
        message = (
            f"PSI={psi_value:.4f} — moderate population shift. "
            "Increase monitoring frequency."
        )
    else:
        severity = "ok"
        alert = False
        message = f"PSI={psi_value:.4f} — distribution is stable."

    _log_alert("data_drift", severity, message)
    return {
        "model": "data_drift",
        "alert": alert,
        "message": message,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# Aggregated alert runner
# ---------------------------------------------------------------------------


def run_all_alerts(monitoring_report: dict) -> list[dict]:
    """Evaluate all alert checks from a unified monitoring report dict.

    Expected keys in ``monitoring_report``
    ----------------------------------------
    forecasting:
        current_mape     float
        baseline_mape    float
        threshold_pct    float  (optional, default 0.20)
    churn:
        current_pr_auc   float
        baseline_pr_auc  float
        threshold_pct    float  (optional, default 0.10)
    drift:
        psi_value        float

    Returns
    -------
    list[dict]
        All alert dicts in order [forecasting, churn, drift].
        Each dict carries 'model', 'alert', 'message', 'severity'.
    """
    alerts: list[dict] = []

    # --- Demand forecasting alert ---
    fc = monitoring_report.get("forecasting", {})
    if fc:
        alerts.append(
            check_forecasting_alert(
                current_mape=fc["current_mape"],
                baseline_mape=fc["baseline_mape"],
                threshold_pct=fc.get("threshold_pct", 0.20),
            )
        )

    # --- Churn model alert ---
    ch = monitoring_report.get("churn", {})
    if ch:
        alerts.append(
            check_churn_alert(
                current_pr_auc=ch["current_pr_auc"],
                baseline_pr_auc=ch["baseline_pr_auc"],
                threshold_pct=ch.get("threshold_pct", 0.10),
            )
        )

    # --- Data drift alert ---
    drift = monitoring_report.get("drift", {})
    if drift:
        alerts.append(check_drift_alert(psi_value=drift["psi_value"]))

    active = [a for a in alerts if a["alert"]]
    logger.info(
        "run_all_alerts: %d total checks, %d active alerts (%d critical, %d warning)",
        len(alerts),
        len(active),
        sum(1 for a in active if a["severity"] == "critical"),
        sum(1 for a in active if a["severity"] == "warning"),
    )
    return alerts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log_alert(model: str, severity: str, message: str) -> None:
    if severity == "critical":
        logger.error("[ALERT][%s][%s] %s", severity.upper(), model, message)
    elif severity == "warning":
        logger.warning("[ALERT][%s][%s] %s", severity.upper(), model, message)
    else:
        logger.info("[ALERT][%s][%s] %s", severity.upper(), model, message)
