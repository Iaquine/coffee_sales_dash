"""
app.py — Entrypoint do Coffee Retail Intelligence Dashboard.

Execução:
    cd coffee_sales/
    python src/dashboard/app.py

Acesso: http://localhost:8050
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Garantir que src/ está no path quando executado como script
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
import pandas as pd

from dashboard.data_loader import (
    load_churn_predictions,
    load_demand_forecast,
    load_mlflow_metrics,
    load_transactions,
)
from dashboard.components.kpi_card import kpi_card
from dashboard.components.charts import forecast_chart, roi_bar_chart
from dashboard.layouts.overview import layout as overview_layout
from dashboard.layouts.demand import layout as demand_layout
from dashboard.layouts.customers import layout as customers_layout
from dashboard.layouts.mlops import layout as mlops_layout

# ── Constantes de estilo ──────────────────────────────────────────────────────
_BROWN = "#3E1C0A"
_BROWN_MID = "#6B3A2A"
_BEIGE = "#D6C5B0"
_CREAM = "#F7F4F0"
_WHITE = "#FFFFFF"

TAB_STYLE = {
    "fontFamily": "-apple-system, 'Segoe UI', system-ui, sans-serif",
    "fontSize": "14px",
    "padding": "10px 20px",
    "color": "#7A6555",
    "borderBottom": f"3px solid transparent",
    "backgroundColor": _CREAM,
}
TAB_SELECTED = {
    **TAB_STYLE,
    "color": _BROWN,
    "fontWeight": "700",
    "borderBottom": f"3px solid {_BROWN_MID}",
    "backgroundColor": _WHITE,
}

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Coffee Retail Intelligence",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # Para deploys WSGI

# ── Layout principal ──────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H3(
                            "☕ Coffee Retail Intelligence Dashboard",
                            style={"color": _BROWN, "fontWeight": "700", "marginBottom": "2px"},
                        ),
                        html.Small(
                            "Visão executiva de vendas, demanda, clientes e ROI dos modelos de ML.",
                            style={"color": "#7A6555"},
                        ),
                    ],
                    style={"padding": "16px 0 8px 0"},
                )
            )
        ),

        html.Hr(style={"borderColor": _BEIGE, "margin": "4px 0 0 0"}),

        # Tabs
        dcc.Tabs(
            id="main-tabs",
            value="tab-overview",
            children=[
                dcc.Tab(label="📊 Visão Geral",      value="tab-overview",   style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="📦 Demanda & Estoque", value="tab-demand",     style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="👥 Clientes & Churn",  value="tab-customers",  style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="⚙️ MLOps & ROI",       value="tab-mlops",      style=TAB_STYLE, selected_style=TAB_SELECTED),
            ],
            style={"marginTop": "8px"},
            colors={"border": _BEIGE, "primary": _BROWN_MID, "background": _CREAM},
        ),

        # Conteúdo da aba selecionada (renderizado por callback)
        html.Div(id="tab-content", style={"marginTop": "8px"}),
    ],
    fluid=True,
    style={"backgroundColor": _WHITE, "minHeight": "100vh", "padding": "0 16px"},
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

# 1. Roteamento de abas
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab: str):
    if tab == "tab-overview":
        return overview_layout
    if tab == "tab-demand":
        return demand_layout
    if tab == "tab-customers":
        return customers_layout
    if tab == "tab-mlops":
        return mlops_layout
    return html.Div("Aba não encontrada.")


# 2. Filtro de data na Aba 1 — atualiza série temporal e KPIs
@app.callback(
    Output("overview-timeseries", "figure"),
    Output("overview-kpi-row", "children"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
)
def update_overview(start_date, end_date):
    from dashboard.components.charts import revenue_timeseries

    df = load_transactions()
    if start_date and end_date:
        mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
        df = df[mask]

    total_rev = df["money"].sum()
    avg_ticket = df["money"].mean() if len(df) > 0 else 0.0
    total_tx = len(df)
    unique_products = df["coffee_name"].nunique()

    kpis = [
        dbc.Col(kpi_card("Receita Total", f"R$ {total_rev:,.2f}", icon="💰"), md=3, xs=6, className="mb-3"),
        dbc.Col(kpi_card("Ticket Médio", f"R$ {avg_ticket:,.2f}", icon="🎫", color="#A0664A"), md=3, xs=6, className="mb-3"),
        dbc.Col(kpi_card("Total Transações", f"{total_tx:,}", icon="🧾", color="#6B3A2A"), md=3, xs=6, className="mb-3"),
        dbc.Col(kpi_card("Produtos Únicos", str(unique_products), icon="☕", color="#3E1C0A"), md=3, xs=6, className="mb-3"),
    ]
    return revenue_timeseries(df), kpis


# 3. Dropdown de produto na Aba 2 — atualiza gráfico e KPIs
@app.callback(
    Output("demand-forecast-chart", "figure"),
    Output("demand-kpi-mape-col", "children"),
    Output("demand-kpi-7d-col", "children"),
    Output("demand-kpi-30d-col", "children"),
    Output("demand-risk-badge", "children"),
    Input("demand-product-dropdown", "value"),
)
def update_demand(product: str):
    df_tx = load_transactions()
    fc = load_demand_forecast()
    metrics = load_mlflow_metrics()

    fig = forecast_chart(df_tx, fc, product)

    # KPIs
    fc_prod = fc[fc["product"] == product]
    fc7_sum = fc_prod[fc_prod["horizon"] == 7]["predicted_revenue"].sum()
    fc30_sum = fc_prod[fc_prod["horizon"] == 30]["predicted_revenue"].sum()
    mape = metrics.get("demand", {}).get("mape", float("nan"))
    mape_str = f"{mape:.1f}%" if not math.isnan(mape) else "—"

    # Histórico dos últimos 60 dias para o produto selecionado
    hist_prod = df_tx[df_tx["coffee_name"] == product].groupby("date")["money"].sum()
    hist_avg = float(hist_prod.mean()) if len(hist_prod) > 0 else 0.0

    # Alerta: se previsão 7d < 80% da média histórica diária × 7
    alert = []
    if hist_avg > 0 and fc7_sum < 0.80 * (hist_avg * 7):
        alert = [
            dbc.Badge(
                "⚠️ Risco de Queda",
                color="danger",
                style={"fontSize": "0.78rem", "padding": "5px 10px"},
            )
        ]

    return (
        fig,
        kpi_card("MAPE do Modelo", mape_str, subtitle="erro percentual médio", icon="📐", color="#6B3A2A"),
        kpi_card("Previsão 7 dias", f"R$ {fc7_sum:,.0f}", icon="📅", color="#A0664A"),
        kpi_card("Previsão 30 dias", f"R$ {fc30_sum:,.0f}", icon="🗓️", color="#3E1C0A"),
        alert,
    )


# 4. Inputs de ROI na Aba 4 — recalculam KPIs de ROI (debounce já no Input)
@app.callback(
    Output("roi-kpi-forecasting-col", "children"),
    Output("roi-kpi-churn-col", "children"),
    Output("roi-kpi-combined-col", "children"),
    Output("roi-kpi-payback-col", "children"),
    Output("roi-bar-chart", "figure"),
    Input("roi-custo-desperdicio", "value"),
    Input("roi-margem", "value"),
    Input("roi-custo-campanha", "value"),
    Input("roi-custo-infra", "value"),
)
def update_roi(custo_desperdicio, margem, custo_campanha, custo_infra):
    # Defaults seguros
    custo_desperdicio = float(custo_desperdicio or 5.0)
    margem = float(margem or 0.35)
    custo_campanha = float(custo_campanha or 10.0)
    custo_infra = float(custo_infra or 500.0)

    churn = load_churn_predictions()
    df_tx = load_transactions()

    # ── Forecasting ROI ────────────────────────────────────────────────────
    # Baseline: previsão sem modelo (média diária × 30d); benefício = redução de desperdício
    daily_avg = df_tx.groupby("date")["money"].sum().mean()
    # Estimativa: modelo reduz desperdício em ~15% do custo diário de estoque
    units_saved_monthly = daily_avg * 0.15 / custo_desperdicio if custo_desperdicio > 0 else 0
    forecast_waste_savings = units_saved_monthly * custo_desperdicio * 30  # mensal
    forecast_total_benefit = forecast_waste_savings
    forecast_cost = custo_infra  # custo mensal de infra
    forecast_net = forecast_total_benefit - forecast_cost
    forecast_roi = (forecast_net / forecast_cost * 100) if forecast_cost > 0 else 0.0

    # ── Churn ROI ──────────────────────────────────────────────────────────
    at_risk = churn[churn["churn_label"] == 1]
    avg_monetary = float(at_risk["monetary"].mean()) if len(at_risk) > 0 else 0.0
    # Assume 30% de retenção com campanha
    retained = len(at_risk) * 0.30
    revenue_saved = retained * avg_monetary * margem
    campaign_cost = len(at_risk) * custo_campanha
    churn_net = revenue_saved - campaign_cost
    churn_roi = (churn_net / campaign_cost * 100) if campaign_cost > 0 else 0.0

    # ── Combinado ──────────────────────────────────────────────────────────
    total_benefit = forecast_total_benefit + revenue_saved
    total_cost = forecast_cost + campaign_cost
    combined_net = total_benefit - total_cost
    combined_roi = (combined_net / total_cost * 100) if total_cost > 0 else 0.0

    # Payback (dias): custo_infra / benefício_diário
    daily_benefit = total_benefit / 30 if total_benefit > 0 else 0
    payback_days = int(custo_infra / daily_benefit) if daily_benefit > 0 else 0

    fig = roi_bar_chart(forecast_total_benefit, forecast_cost, revenue_saved, campaign_cost)

    def _roi_str(v: float) -> str:
        return f"{v:+.1f}%"

    return (
        kpi_card("ROI Forecasting", _roi_str(forecast_roi), subtitle=f"Benefício R$ {forecast_total_benefit:,.0f}", icon="📦", color="#6B3A2A"),
        kpi_card("ROI Churn", _roi_str(churn_roi), subtitle=f"Receita salva R$ {revenue_saved:,.0f}", icon="🔄", color="#A0664A"),
        kpi_card("ROI Combinado", _roi_str(combined_roi), subtitle=f"Líquido R$ {combined_net:,.0f}", icon="💼", color="#3E1C0A"),
        kpi_card("Payback", f"{payback_days}d", subtitle="baseado no custo de infra", icon="📆", color="#7A6555"),
        fig,
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
