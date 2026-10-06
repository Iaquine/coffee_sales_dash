"""
roi_calculator.py — Financial ROI calculations for production ML models.

ALL financial parameters are received as function arguments.
Zero hardcoded business constants — this module is designed to be driven
by user inputs (e.g. from the Dash dashboard, a config file, or CLI).

Public API
----------
calculate_forecasting_roi(
    mape_baseline, mape_model,
    avg_daily_revenue, waste_rate, stockout_rate, margin,
    infra_cost_monthly
) -> dict

calculate_churn_roi(
    n_churners_detected, retention_rate,
    avg_monthly_revenue_per_customer, margin,
    campaign_cost_per_customer, infra_cost_monthly
) -> dict

calculate_total_roi(forecasting_roi, churn_roi) -> dict
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demand forecasting ROI
# ---------------------------------------------------------------------------


def calculate_forecasting_roi(
    mape_baseline: float,
    mape_model: float,
    avg_daily_revenue: float,
    waste_rate: float,
    stockout_rate: float,
    margin: float,
    infra_cost_monthly: float,
) -> dict:
    """Calculate monthly ROI from the demand-forecasting model.

    Parameters
    ----------
    mape_baseline : float
        MAPE of the naive/baseline forecast (e.g. 0.35 = 35 %).
    mape_model : float
        MAPE of the production XGBoost model (e.g. 0.12 = 12 %).
    avg_daily_revenue : float
        Average gross daily revenue (same currency as infra_cost_monthly).
    waste_rate : float
        Fraction of daily revenue lost to spoilage/excess inventory under the
        baseline (e.g. 0.08 = 8 %).
    stockout_rate : float
        Fraction of potential daily revenue lost to stockouts under the
        baseline (e.g. 0.05 = 5 %).
    margin : float
        Gross margin on recovered revenue (e.g. 0.60 = 60 %).
    infra_cost_monthly : float
        Monthly infrastructure + maintenance cost for the model.

    Returns
    -------
    dict with keys:
        waste_savings       — monthly reduction in spoilage cost
        stockout_recovery   — monthly recovered margin from fewer stockouts
        total_benefit       — waste_savings + stockout_recovery
        total_cost          — infra_cost_monthly
        roi_pct             — (total_benefit - total_cost) / total_cost * 100
        payback_days        — days until cumulative benefit covers total_cost
    """
    avg_monthly_revenue = avg_daily_revenue * 30

    # Improvement fraction: how much better is the model vs baseline?
    mape_improvement = max(0.0, mape_baseline - mape_model)
    improvement_ratio = mape_improvement / mape_baseline if mape_baseline > 0 else 0.0

    waste_savings = avg_monthly_revenue * waste_rate * improvement_ratio
    stockout_recovery = avg_monthly_revenue * stockout_rate * improvement_ratio * margin

    total_benefit = waste_savings + stockout_recovery
    total_cost = infra_cost_monthly

    roi_pct = (
        (total_benefit - total_cost) / total_cost * 100 if total_cost > 0 else 0.0
    )

    # Payback: days for cumulative daily benefit to cover monthly cost
    daily_benefit = total_benefit / 30 if total_benefit > 0 else 0.0
    payback_days = total_cost / daily_benefit if daily_benefit > 0 else float("inf")

    result = {
        "waste_savings": round(waste_savings, 2),
        "stockout_recovery": round(stockout_recovery, 2),
        "total_benefit": round(total_benefit, 2),
        "total_cost": round(total_cost, 2),
        "roi_pct": round(roi_pct, 2),
        "payback_days": round(payback_days, 1) if payback_days != float("inf") else None,
    }

    logger.info(
        "Forecasting ROI | benefit=%.2f cost=%.2f ROI=%.1f%% payback=%.1f days",
        total_benefit,
        total_cost,
        roi_pct,
        payback_days if payback_days != float("inf") else -1,
    )
    return result


# ---------------------------------------------------------------------------
# Churn prevention ROI
# ---------------------------------------------------------------------------


def calculate_churn_roi(
    n_churners_detected: int,
    retention_rate: float,
    avg_monthly_revenue_per_customer: float,
    margin: float,
    campaign_cost_per_customer: float,
    infra_cost_monthly: float,
) -> dict:
    """Calculate monthly ROI from the churn-prevention model.

    Parameters
    ----------
    n_churners_detected : int
        Number of at-risk customers identified by the model this month.
    retention_rate : float
        Fraction of detected churners successfully retained after campaign
        (e.g. 0.30 = 30 %).
    avg_monthly_revenue_per_customer : float
        Average monthly gross revenue per retained customer.
    margin : float
        Gross margin applied to retained revenue.
    campaign_cost_per_customer : float
        Cost to execute a retention campaign for one customer.
    infra_cost_monthly : float
        Monthly infrastructure + maintenance cost for the model.

    Returns
    -------
    dict with keys:
        revenue_saved     — monthly margin recovered from retained customers
        campaign_cost     — total campaign spend
        total_cost        — campaign_cost + infra_cost_monthly
        net_benefit       — revenue_saved - total_cost
        roi_pct           — net_benefit / total_cost * 100
    """
    customers_retained = n_churners_detected * retention_rate
    revenue_saved = customers_retained * avg_monthly_revenue_per_customer * margin
    campaign_cost = n_churners_detected * campaign_cost_per_customer
    total_cost = campaign_cost + infra_cost_monthly
    net_benefit = revenue_saved - total_cost

    roi_pct = (net_benefit / total_cost * 100) if total_cost > 0 else 0.0

    result = {
        "customers_retained": round(customers_retained, 1),
        "revenue_saved": round(revenue_saved, 2),
        "campaign_cost": round(campaign_cost, 2),
        "total_cost": round(total_cost, 2),
        "net_benefit": round(net_benefit, 2),
        "roi_pct": round(roi_pct, 2),
    }

    logger.info(
        "Churn ROI | retained=%.1f revenue_saved=%.2f net_benefit=%.2f ROI=%.1f%%",
        customers_retained,
        revenue_saved,
        net_benefit,
        roi_pct,
    )
    return result


# ---------------------------------------------------------------------------
# Combined ROI summary
# ---------------------------------------------------------------------------


def calculate_total_roi(
    forecasting_roi: dict,
    churn_roi: dict,
) -> dict:
    """Combine forecasting and churn ROI into an overall project summary.

    Parameters
    ----------
    forecasting_roi : dict
        Output of ``calculate_forecasting_roi``.
    churn_roi : dict
        Output of ``calculate_churn_roi``.

    Returns
    -------
    dict with keys:
        total_net_benefit   — sum of both net benefits
        total_cost          — sum of both total costs
        combined_roi_pct    — total_net_benefit / total_cost * 100
    """
    forecasting_net = forecasting_roi["total_benefit"] - forecasting_roi["total_cost"]
    churn_net = churn_roi["net_benefit"]

    total_net_benefit = forecasting_net + churn_net
    total_cost = forecasting_roi["total_cost"] + churn_roi["total_cost"]
    combined_roi_pct = (
        (total_net_benefit / total_cost * 100) if total_cost > 0 else 0.0
    )

    result = {
        "total_net_benefit": round(total_net_benefit, 2),
        "total_cost": round(total_cost, 2),
        "combined_roi_pct": round(combined_roi_pct, 2),
    }

    logger.info(
        "Total ROI | net_benefit=%.2f total_cost=%.2f combined_roi=%.1f%%",
        total_net_benefit,
        total_cost,
        combined_roi_pct,
    )
    return result
