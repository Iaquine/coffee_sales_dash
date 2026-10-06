"""
test_monitoring.py — Unit tests for the src/monitoring package.

Coverage:
  - compute_psi: identical distributions (PSI ≈ 0) and very different distributions (PSI > 0.2)
  - ks_test: identical distributions (high p_value) and different distributions (low p_value)
  - detect_data_drift: drift_detected flag behaviour
  - detect_concept_drift: no drift / drift scenarios
  - calculate_forecasting_roi: correct roi_pct formula
  - calculate_churn_roi: correct net_benefit formula
  - calculate_total_roi: combined_roi_pct aggregation
  - check_drift_alert: all three severity levels (ok / warning / critical)
  - check_forecasting_alert: ok / warning / critical
  - check_churn_alert: ok / critical
  - run_all_alerts: integration through a full monitoring report
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root (coffee_sales/) is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.alerts import (
    check_churn_alert,
    check_drift_alert,
    check_forecasting_alert,
    run_all_alerts,
)
from src.monitoring.drift_detector import (
    compute_psi,
    detect_concept_drift,
    detect_data_drift,
    ks_test,
)
from src.monitoring.roi_calculator import (
    calculate_churn_roi,
    calculate_forecasting_roi,
    calculate_total_roi,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def identical_dist() -> tuple[np.ndarray, np.ndarray]:
    """Two samples drawn from the exact same distribution (Normal μ=0 σ=1)."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1_000)
    b = rng.normal(0, 1, 1_000)
    return a, b


@pytest.fixture()
def different_dist() -> tuple[np.ndarray, np.ndarray]:
    """Two samples from very different distributions (Normal μ=0 vs μ=5)."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1_000)
    b = rng.normal(5, 1, 1_000)
    return a, b


# ===========================================================================
# compute_psi
# ===========================================================================


class TestComputePsi:
    def test_identical_distributions_near_zero(self, identical_dist):
        a, b = identical_dist
        psi = compute_psi(a, b)
        # PSI of two samples from the same distribution should be very small
        assert psi < 0.05, f"Expected PSI ≈ 0 for identical distributions, got {psi:.4f}"

    def test_very_different_distributions_above_threshold(self, different_dist):
        a, b = different_dist
        psi = compute_psi(a, b)
        # PSI of distributions 5 sigma apart should be clearly > 0.2
        assert psi > 0.2, f"Expected PSI > 0.2 for very different distributions, got {psi:.4f}"

    def test_exact_same_array_is_zero(self):
        arr = np.linspace(0, 10, 500)
        psi = compute_psi(arr, arr)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_returns_float(self, identical_dist):
        a, b = identical_dist
        assert isinstance(compute_psi(a, b), float)

    def test_accepts_pandas_series(self, identical_dist):
        a, b = identical_dist
        psi = compute_psi(pd.Series(a), pd.Series(b))
        assert isinstance(psi, float)
        assert psi >= 0.0


# ===========================================================================
# ks_test
# ===========================================================================


class TestKsTest:
    def test_identical_high_p_value(self, identical_dist):
        a, b = identical_dist
        result = ks_test(a, b)
        assert "statistic" in result
        assert "p_value" in result
        # With large identical samples the statistic should be small
        assert result["statistic"] < 0.1

    def test_different_low_p_value(self, different_dist):
        a, b = different_dist
        result = ks_test(a, b)
        # Vastly different distributions → statistic near 1, p_value ≈ 0
        assert result["statistic"] > 0.8
        assert result["p_value"] < 1e-6

    def test_result_keys(self, identical_dist):
        a, b = identical_dist
        result = ks_test(a, b)
        assert set(result.keys()) == {"statistic", "p_value"}


# ===========================================================================
# detect_data_drift
# ===========================================================================


class TestDetectDataDrift:
    def _make_df(self, data_dict: dict) -> pd.DataFrame:
        return pd.DataFrame(data_dict)

    def test_no_drift_when_identical(self, identical_dist):
        a, b = identical_dist
        df_ref = self._make_df({"x": a})
        df_cur = self._make_df({"x": b})
        result = detect_data_drift(df_ref, df_cur, features=["x"])
        assert not result["drift_detected"]

    def test_drift_detected_when_very_different(self, different_dist):
        a, b = different_dist
        df_ref = self._make_df({"x": a})
        df_cur = self._make_df({"x": b})
        result = detect_data_drift(df_ref, df_cur, features=["x"])
        assert result["drift_detected"]

    def test_feature_keys_present(self, identical_dist):
        a, b = identical_dist
        df_ref = self._make_df({"feat1": a, "feat2": a})
        df_cur = self._make_df({"feat1": b, "feat2": b})
        result = detect_data_drift(df_ref, df_cur, features=["feat1", "feat2"])
        assert "feat1" in result["features"]
        assert "feat2" in result["features"]

    def test_missing_feature_skipped_gracefully(self, identical_dist):
        a, b = identical_dist
        df_ref = self._make_df({"x": a})
        df_cur = self._make_df({"x": b})
        # 'missing_col' is not in either DataFrame
        result = detect_data_drift(df_ref, df_cur, features=["x", "missing_col"])
        assert "x" in result["features"]
        assert "missing_col" not in result["features"]


# ===========================================================================
# detect_concept_drift
# ===========================================================================


class TestDetectConceptDrift:
    def test_no_drift_stable_history(self):
        history = [{"mape": 0.15 + i * 0.001} for i in range(12)]
        result = detect_concept_drift(history, "mape", threshold_pct=0.20)
        # Only ~7% drift (from 0.15 to 0.161) — well below 20%
        assert not result["drift_detected"]

    def test_drift_large_degradation(self):
        # Start at 0.10, end at 0.20 → 100% increase
        stable = [{"mape": 0.10}] * 6
        degraded = [{"mape": 0.20}] * 6
        result = detect_concept_drift(stable + degraded, "mape", threshold_pct=0.20)
        assert result["drift_detected"]
        assert result["current_value"] == pytest.approx(0.20)

    def test_empty_history_returns_no_drift(self):
        result = detect_concept_drift([], "mape")
        assert not result["drift_detected"]
        assert result["baseline_value"] is None

    def test_unknown_metric_key_raises(self):
        history = [{"mape": 0.10}]
        with pytest.raises(KeyError):
            detect_concept_drift(history, "nonexistent_key")


# ===========================================================================
# calculate_forecasting_roi
# ===========================================================================


class TestCalculateForecastingRoi:
    def _base_params(self) -> dict:
        return {
            "mape_baseline": 0.35,
            "mape_model": 0.14,
            "avg_daily_revenue": 1_500.0,
            "waste_rate": 0.08,
            "stockout_rate": 0.05,
            "margin": 0.60,
            "infra_cost_monthly": 300.0,
        }

    def test_roi_pct_is_positive_when_benefit_exceeds_cost(self):
        result = calculate_forecasting_roi(**self._base_params())
        assert result["roi_pct"] > 0

    def test_roi_pct_formula_exact(self):
        """ROI% = (total_benefit - total_cost) / total_cost * 100."""
        result = calculate_forecasting_roi(**self._base_params())
        expected_roi = (
            (result["total_benefit"] - result["total_cost"]) / result["total_cost"] * 100
        )
        assert result["roi_pct"] == pytest.approx(expected_roi, rel=1e-4)

    def test_total_benefit_is_waste_plus_stockout(self):
        result = calculate_forecasting_roi(**self._base_params())
        expected = result["waste_savings"] + result["stockout_recovery"]
        assert result["total_benefit"] == pytest.approx(expected, abs=0.01)

    def test_no_benefit_when_model_equal_to_baseline(self):
        params = self._base_params()
        params["mape_model"] = params["mape_baseline"]
        result = calculate_forecasting_roi(**params)
        assert result["waste_savings"] == pytest.approx(0.0, abs=0.01)
        assert result["total_benefit"] == pytest.approx(0.0, abs=0.01)

    def test_payback_days_none_when_no_benefit(self):
        params = self._base_params()
        params["mape_model"] = params["mape_baseline"]
        result = calculate_forecasting_roi(**params)
        assert result["payback_days"] is None

    def test_result_keys(self):
        result = calculate_forecasting_roi(**self._base_params())
        expected_keys = {
            "waste_savings", "stockout_recovery", "total_benefit",
            "total_cost", "roi_pct", "payback_days",
        }
        assert set(result.keys()) == expected_keys


# ===========================================================================
# calculate_churn_roi
# ===========================================================================


class TestCalculateChurnRoi:
    def _base_params(self) -> dict:
        return {
            "n_churners_detected": 28,
            "retention_rate": 0.30,
            "avg_monthly_revenue_per_customer": 120.0,
            "margin": 0.60,
            "campaign_cost_per_customer": 15.0,
            "infra_cost_monthly": 150.0,
        }

    def test_net_benefit_formula(self):
        """net_benefit = revenue_saved - total_cost."""
        result = calculate_churn_roi(**self._base_params())
        expected_net = result["revenue_saved"] - result["total_cost"]
        assert result["net_benefit"] == pytest.approx(expected_net, abs=0.01)

    def test_revenue_saved_formula(self):
        """revenue_saved = n * retention_rate * avg_revenue * margin."""
        p = self._base_params()
        result = calculate_churn_roi(**p)
        expected = (
            p["n_churners_detected"]
            * p["retention_rate"]
            * p["avg_monthly_revenue_per_customer"]
            * p["margin"]
        )
        assert result["revenue_saved"] == pytest.approx(expected, abs=0.01)

    def test_roi_pct_formula(self):
        result = calculate_churn_roi(**self._base_params())
        # Recompute from rounded result values; abs tolerance 0.02 to absorb rounding
        expected_roi = result["net_benefit"] / result["total_cost"] * 100
        assert result["roi_pct"] == pytest.approx(expected_roi, abs=0.02)

    def test_zero_churners_yields_zero_benefit(self):
        params = self._base_params()
        params["n_churners_detected"] = 0
        result = calculate_churn_roi(**params)
        assert result["revenue_saved"] == pytest.approx(0.0)
        assert result["customers_retained"] == pytest.approx(0.0)

    def test_result_keys(self):
        result = calculate_churn_roi(**self._base_params())
        expected_keys = {
            "customers_retained", "revenue_saved", "campaign_cost",
            "total_cost", "net_benefit", "roi_pct",
        }
        assert set(result.keys()) == expected_keys


# ===========================================================================
# calculate_total_roi
# ===========================================================================


class TestCalculateTotalRoi:
    def _make_rois(self):
        forecasting = calculate_forecasting_roi(
            mape_baseline=0.35, mape_model=0.14,
            avg_daily_revenue=1_500.0, waste_rate=0.08,
            stockout_rate=0.05, margin=0.60, infra_cost_monthly=300.0,
        )
        churn = calculate_churn_roi(
            n_churners_detected=28, retention_rate=0.30,
            avg_monthly_revenue_per_customer=120.0, margin=0.60,
            campaign_cost_per_customer=15.0, infra_cost_monthly=150.0,
        )
        return forecasting, churn

    def test_combined_roi_pct_formula(self):
        f, c = self._make_rois()
        result = calculate_total_roi(f, c)
        expected = result["total_net_benefit"] / result["total_cost"] * 100
        assert result["combined_roi_pct"] == pytest.approx(expected, rel=1e-4)

    def test_total_cost_is_sum(self):
        f, c = self._make_rois()
        result = calculate_total_roi(f, c)
        assert result["total_cost"] == pytest.approx(f["total_cost"] + c["total_cost"], abs=0.01)

    def test_result_keys(self):
        f, c = self._make_rois()
        result = calculate_total_roi(f, c)
        assert set(result.keys()) == {"total_net_benefit", "total_cost", "combined_roi_pct"}


# ===========================================================================
# check_drift_alert
# ===========================================================================


class TestCheckDriftAlert:
    def test_ok_when_psi_below_warning(self):
        result = check_drift_alert(0.05)
        assert result["severity"] == "ok"
        assert not result["alert"]

    def test_warning_when_psi_between_thresholds(self):
        result = check_drift_alert(0.15)
        assert result["severity"] == "warning"
        assert result["alert"]

    def test_critical_when_psi_above_critical(self):
        result = check_drift_alert(0.25)
        assert result["severity"] == "critical"
        assert result["alert"]

    def test_boundary_psi_01_is_warning(self):
        result = check_drift_alert(0.10)
        assert result["severity"] == "warning"

    def test_boundary_psi_02_is_critical(self):
        result = check_drift_alert(0.20)
        assert result["severity"] == "critical"

    def test_result_keys(self):
        result = check_drift_alert(0.05)
        assert set(result.keys()) == {"model", "alert", "message", "severity"}


# ===========================================================================
# check_forecasting_alert
# ===========================================================================


class TestCheckForecastingAlert:
    def test_ok_when_model_is_better(self):
        result = check_forecasting_alert(current_mape=0.12, baseline_mape=0.35)
        assert result["severity"] == "ok"
        assert not result["alert"]

    def test_critical_when_large_degradation(self):
        result = check_forecasting_alert(current_mape=0.50, baseline_mape=0.20)
        assert result["severity"] == "critical"
        assert result["alert"]

    def test_warning_when_moderate_degradation(self):
        # 15% relative increase — above 50% of threshold (20%), below threshold
        result = check_forecasting_alert(
            current_mape=0.23, baseline_mape=0.20, threshold_pct=0.20
        )
        assert result["severity"] == "warning"

    def test_zero_baseline_returns_ok(self):
        result = check_forecasting_alert(current_mape=0.20, baseline_mape=0.0)
        assert result["severity"] == "ok"


# ===========================================================================
# check_churn_alert
# ===========================================================================


class TestCheckChurnAlert:
    def test_ok_when_stable(self):
        result = check_churn_alert(current_pr_auc=0.75, baseline_pr_auc=0.75)
        assert result["severity"] == "ok"
        assert not result["alert"]

    def test_critical_when_large_drop(self):
        result = check_churn_alert(current_pr_auc=0.50, baseline_pr_auc=0.75)
        assert result["severity"] == "critical"
        assert result["alert"]


# ===========================================================================
# run_all_alerts (integration)
# ===========================================================================


class TestRunAllAlerts:
    def _ok_report(self) -> dict:
        return {
            "forecasting": {"baseline_mape": 0.35, "current_mape": 0.14},
            "churn": {"baseline_pr_auc": 0.75, "current_pr_auc": 0.75},
            "drift": {"psi_value": 0.05},
        }

    def test_returns_three_alert_dicts(self):
        alerts = run_all_alerts(self._ok_report())
        assert len(alerts) == 3

    def test_all_ok_when_healthy(self):
        alerts = run_all_alerts(self._ok_report())
        assert all(a["severity"] == "ok" for a in alerts)

    def test_critical_alert_present_when_drift_high(self):
        report = self._ok_report()
        report["drift"]["psi_value"] = 0.35
        alerts = run_all_alerts(report)
        severities = {a["model"]: a["severity"] for a in alerts}
        assert severities["data_drift"] == "critical"

    def test_empty_report_returns_empty_list(self):
        alerts = run_all_alerts({})
        assert alerts == []

    def test_partial_report_skips_missing_sections(self):
        report = {"forecasting": {"baseline_mape": 0.35, "current_mape": 0.14}}
        alerts = run_all_alerts(report)
        assert len(alerts) == 1
        assert alerts[0]["model"] == "demand_forecast_xgboost"
