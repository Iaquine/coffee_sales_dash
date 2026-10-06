"""
overview.py — Aba 1: Visão Geral

Contém o layout estático e é alimentada por callbacks registrados em app.py.
"""

from __future__ import annotations

import pandas as pd
from dash import dcc, html
import dash_bootstrap_components as dbc

from dashboard.data_loader import load_transactions
from dashboard.components.kpi_card import kpi_card
from dashboard.components.charts import revenue_timeseries, revenue_by_product, sales_heatmap

# ── Valores iniciais (carregados na inicialização) ────────────────────────────
_df = load_transactions()
_total_revenue = _df["money"].sum()
_avg_ticket = _df["money"].mean()
_total_tx = len(_df)
_unique_products = _df["coffee_name"].nunique()
_min_date = _df["date"].min()
_max_date = _df["date"].max()


def _fmt_brl(value: float) -> str:
    return f"R$ {value:,.2f}"


# ── Layout ────────────────────────────────────────────────────────────────────
layout = dbc.Container(
    [
        # Título da aba + insight executivo
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H5("Visão Geral de Vendas", className="mb-0", style={"color": "#3E1C0A"}),
                    html.Small(
                        "Receita agregada, tendência e distribuição por produto e horário.",
                        style={"color": "#7A6555"},
                    ),
                ]),
                className="mb-3 mt-2",
            )
        ),

        # KPI cards
        dbc.Row(
            [
                dbc.Col(kpi_card("Receita Total", _fmt_brl(_total_revenue), icon="💰"), md=3, xs=6, className="mb-3"),
                dbc.Col(kpi_card("Ticket Médio", _fmt_brl(_avg_ticket), icon="🎫", color="#A0664A"), md=3, xs=6, className="mb-3"),
                dbc.Col(kpi_card("Total Transações", f"{_total_tx:,}", icon="🧾", color="#6B3A2A"), md=3, xs=6, className="mb-3"),
                dbc.Col(kpi_card("Produtos Únicos", str(_unique_products), icon="☕", color="#3E1C0A"), md=3, xs=6, className="mb-3"),
            ],
            id="overview-kpi-row",
        ),

        # Filtro de período
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(html.Label("Período:", style={"color": "#7A6555", "fontWeight": "600"}), width="auto", className="align-self-center"),
                                dbc.Col(
                                    dcc.DatePickerRange(
                                        id="overview-date-range",
                                        min_date_allowed=str(_min_date.date()),
                                        max_date_allowed=str(_max_date.date()),
                                        start_date=str(_min_date.date()),
                                        end_date=str(_max_date.date()),
                                        display_format="DD/MM/YYYY",
                                        style={"fontSize": "13px"},
                                    ),
                                    width="auto",
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

        # Série temporal
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id="overview-timeseries",
                    figure=revenue_timeseries(_df),
                    config={"displayModeBar": False},
                ),
                className="mb-3",
            )
        ),

        # Barras por produto + Heatmap
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="overview-by-product",
                        figure=revenue_by_product(_df),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                    className="mb-3",
                ),
                dbc.Col(
                    dcc.Graph(
                        id="overview-heatmap",
                        figure=sales_heatmap(_df),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                    className="mb-3",
                ),
            ]
        ),
    ],
    fluid=True,
    className="py-2",
)
