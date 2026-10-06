"""
demand.py — Aba 2: Demanda & Estoque

Layout com dropdown de produto, gráfico de previsão e KPIs do modelo.
Callbacks registrados em app.py.
"""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from dashboard.data_loader import load_demand_forecast, load_mlflow_metrics
from dashboard.components.kpi_card import kpi_card
from dashboard.components.charts import forecast_chart

# ── Dados iniciais ────────────────────────────────────────────────────────────
_forecast = load_demand_forecast()
_products = sorted(_forecast["product"].unique().tolist())
_default_product = _products[0]
_metrics = load_mlflow_metrics()

# KPIs iniciais para o produto padrão
def _initial_kpis(product: str) -> tuple[str, str, str]:
    fc = _forecast[_forecast["product"] == product]
    fc7 = fc[fc["horizon"] == 7]["predicted_revenue"].sum()
    fc30 = fc[fc["horizon"] == 30]["predicted_revenue"].sum()
    mape = _metrics.get("demand", {}).get("mape", float("nan"))
    return (
        f"{mape:.1f}%" if not __import__("math").isnan(mape) else "—",
        f"R$ {fc7:,.0f}",
        f"R$ {fc30:,.0f}",
    )


_mape_str, _fc7_str, _fc30_str = _initial_kpis(_default_product)

# ── Layout ────────────────────────────────────────────────────────────────────
layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H5("Demanda & Estoque", className="mb-0", style={"color": "#3E1C0A"}),
                    html.Small(
                        "Previsões de curto prazo por produto com intervalo de confiança.",
                        style={"color": "#7A6555"},
                    ),
                ]),
                className="mb-3 mt-2",
            )
        ),

        # Dropdown de produto
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Label("Produto:", style={"color": "#7A6555", "fontWeight": "600"}),
                                    width="auto",
                                    className="align-self-center",
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="demand-product-dropdown",
                                        options=[{"label": p, "value": p} for p in _products],
                                        value=_default_product,
                                        clearable=False,
                                        style={"minWidth": "220px"},
                                    ),
                                    width="auto",
                                ),
                                # Badge de alerta (visibilidade controlada por callback)
                                dbc.Col(
                                    html.Div(
                                        id="demand-risk-badge",
                                        children=[],
                                    ),
                                    width="auto",
                                    className="align-self-center",
                                ),
                            ],
                            align="center",
                        )
                    ),
                    style={"backgroundColor": "#F7F4F0", "border": "1px solid #D6C5B0"},
                ),
                className="mb-3",
            )
        ),

        # KPI cards do modelo
        dbc.Row(
            [
                dbc.Col(
                    kpi_card("MAPE do Modelo", _mape_str, subtitle="Erro percentual médio absoluto", icon="📐", color="#6B3A2A"),
                    md=4, xs=12, className="mb-3",
                    id="demand-kpi-mape-col",
                ),
                dbc.Col(
                    kpi_card("Previsão 7 dias", _fc7_str, icon="📅", color="#A0664A"),
                    md=4, xs=12, className="mb-3",
                    id="demand-kpi-7d-col",
                ),
                dbc.Col(
                    kpi_card("Previsão 30 dias", _fc30_str, icon="🗓️", color="#3E1C0A"),
                    md=4, xs=12, className="mb-3",
                    id="demand-kpi-30d-col",
                ),
            ],
            id="demand-kpi-row",
        ),

        # Gráfico principal de previsão
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id="demand-forecast-chart",
                    figure=forecast_chart(
                        __import__("dashboard.data_loader", fromlist=["load_transactions"]).load_transactions(),
                        _forecast,
                        _default_product,
                    ),
                    config={"displayModeBar": False},
                ),
                className="mb-3",
            )
        ),
    ],
    fluid=True,
    className="py-2",
)
