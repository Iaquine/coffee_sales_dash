"""
customers.py — Aba 3: Clientes & Churn

Layout com scatter de segmentos, donut de distribuição, tabela de alto risco
e KPIs de churn.
"""

from __future__ import annotations

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from dashboard.data_loader import load_churn_predictions, load_customer_segments
from dashboard.components.kpi_card import kpi_card
from dashboard.components.charts import segment_scatter, segment_donut

# ── Dados iniciais ────────────────────────────────────────────────────────────
_churn = load_churn_predictions()
_segments = load_customer_segments()

_at_risk_count = int((_churn["churn_label"] == 1).sum())
_revenue_at_risk = float(_churn.loc[_churn["churn_label"] == 1, "monetary"].sum())
_pct_at_risk = _at_risk_count / len(_churn) * 100 if len(_churn) > 0 else 0.0

# Tabela top-20 alto risco
_top20 = (
    _churn.nlargest(20, "churn_probability")[["card", "segment_name", "churn_probability", "monetary"]]
    .copy()
)
_top20["card_display"] = _top20["card"].apply(lambda c: f"****{str(c)[-4:]}")
_top20["churn_pct"] = (_top20["churn_probability"] * 100).round(1)
_top20["monetary"] = _top20["monetary"].round(2)

_table_data = _top20[["card_display", "segment_name", "churn_pct", "monetary"]].rename(
    columns={
        "card_display": "Cartão",
        "segment_name": "Segmento",
        "churn_pct": "Prob. Churn (%)",
        "monetary": "Receita Total (R$)",
    }
).to_dict("records")

# ── Layout ────────────────────────────────────────────────────────────────────
layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H5("Clientes & Churn", className="mb-0", style={"color": "#3E1C0A"}),
                    html.Small(
                        "Segmentação RFM e identificação de clientes em risco de abandono.",
                        style={"color": "#7A6555"},
                    ),
                ]),
                className="mb-3 mt-2",
            )
        ),

        # KPI cards de churn
        dbc.Row(
            [
                dbc.Col(
                    kpi_card(
                        "Clientes em Risco",
                        f"{_at_risk_count:,}",
                        subtitle="churn_label = 1",
                        icon="⚠️",
                        color="#c62828",
                    ),
                    md=4, xs=12, className="mb-3",
                ),
                dbc.Col(
                    kpi_card(
                        "Receita em Risco",
                        f"R$ {_revenue_at_risk:,.2f}",
                        subtitle="soma do monetary dos clientes em risco",
                        icon="💸",
                        color="#e65100",
                    ),
                    md=4, xs=12, className="mb-3",
                ),
                dbc.Col(
                    kpi_card(
                        "% da Base em Risco",
                        f"{_pct_at_risk:.1f}%",
                        subtitle=f"de {len(_churn):,} clientes identificados",
                        icon="📊",
                        color="#6B3A2A",
                    ),
                    md=4, xs=12, className="mb-3",
                ),
            ]
        ),

        # Scatter + Donut
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="customers-scatter",
                        figure=segment_scatter(_segments),
                        config={"displayModeBar": False},
                    ),
                    md=8, className="mb-3",
                ),
                dbc.Col(
                    dcc.Graph(
                        id="customers-donut",
                        figure=segment_donut(_segments),
                        config={"displayModeBar": False},
                    ),
                    md=4, className="mb-3",
                ),
            ]
        ),

        # Tabela alto risco
        dbc.Row(
            dbc.Col(
                [
                    html.H6(
                        "Top 20 Clientes com Maior Probabilidade de Churn",
                        style={"color": "#3E1C0A", "marginBottom": "8px"},
                    ),
                    html.Small(
                        "🔴 Vermelho: probabilidade > 70%  |  Ação recomendada: acionar campanha de retenção para clientes com churn > 80%.",
                        style={"color": "#7A6555", "display": "block", "marginBottom": "8px"},
                    ),
                    dash_table.DataTable(
                        id="customers-churn-table",
                        data=_table_data,
                        columns=[{"name": c, "id": c} for c in _table_data[0].keys()] if _table_data else [],
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#F7F4F0",
                            "fontWeight": "bold",
                            "color": "#3E1C0A",
                            "borderBottom": "2px solid #D6C5B0",
                        },
                        style_cell={
                            "fontFamily": "-apple-system, 'Segoe UI', sans-serif",
                            "fontSize": "13px",
                            "padding": "8px 12px",
                            "textAlign": "left",
                            "color": "#2C1A0E",
                        },
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": "{Prob. Churn (%)} > 70",
                                    "column_id": "Prob. Churn (%)",
                                },
                                "backgroundColor": "#ffebee",
                                "color": "#c62828",
                                "fontWeight": "600",
                            },
                            {
                                "if": {"filter_query": "{Prob. Churn (%)} > 70"},
                                "backgroundColor": "#fff8f8",
                            },
                            {"if": {"row_index": "odd"}, "backgroundColor": "#fdfaf8"},
                        ],
                        page_size=20,
                        sort_action="native",
                    ),
                ],
                className="mb-3",
            )
        ),
    ],
    fluid=True,
    className="py-2",
)
