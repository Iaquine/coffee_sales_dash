"""
mlops.py — Aba 4: MLOps & ROI

Painel de inputs do usuário + KPIs de ROI calculados por callback em app.py.
"""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from dashboard.data_loader import load_mlflow_metrics
from dashboard.components.kpi_card import kpi_card
from dashboard.components.charts import roi_bar_chart, mape_history_chart

# ── Dados iniciais ────────────────────────────────────────────────────────────
_metrics = load_mlflow_metrics()
_demand_m = _metrics.get("demand", {})
_churn_m = _metrics.get("churn", {})
_clustering_m = _metrics.get("clustering", {})
_monitoring_m = _metrics.get("monitoring", {})
_history = _metrics.get("demand_history", [])

# Status dos modelos
_model_status_data = [
    {
        "Modelo": "Demand Forecasting (XGBoost)",
        "Métrica": "MAPE",
        "Valor": f"{_demand_m.get('mape', float('nan')):.2f}%",
        "Status": "OK" if _demand_m.get("mape", 100) < 50 else "Warning",
    },
    {
        "Modelo": "Churn Classifier (Random Forest)",
        "Métrica": "PR-AUC",
        "Valor": f"{_churn_m.get('pr_auc', 0):.3f}",
        "Status": "OK" if _churn_m.get("pr_auc", 0) >= 0.7 else "Critical",
    },
    {
        "Modelo": "RFM Clustering (K-Means)",
        "Métrica": "Silhouette",
        "Valor": f"{_clustering_m.get('silhouette', 0):.3f}",
        "Status": "OK" if _clustering_m.get("silhouette", 0) >= 0.4 else "Warning",
    },
]

_drift_detected = bool(_monitoring_m.get("drift_detected", 0))
_concept_drift = bool(_monitoring_m.get("concept_drift_detected", 0))

# ── Helpers ───────────────────────────────────────────────────────────────────
def _status_badge(status: str) -> html.Span:
    colors = {"OK": "#2e7d32", "Warning": "#e65100", "Critical": "#c62828"}
    return html.Span(
        status,
        style={
            "backgroundColor": colors.get(status, "#7A6555"),
            "color": "white",
            "padding": "2px 10px",
            "borderRadius": "12px",
            "fontSize": "0.72rem",
            "fontWeight": "600",
        },
    )


def _model_status_table(rows: list[dict]) -> dbc.Table:
    header = html.Thead(html.Tr([html.Th(k) for k in rows[0].keys()]))
    body = html.Tbody([
        html.Tr([
            html.Td(row["Modelo"]),
            html.Td(row["Métrica"]),
            html.Td(row["Valor"]),
            html.Td(_status_badge(row["Status"])),
        ])
        for row in rows
    ])
    return dbc.Table(
        [header, body],
        bordered=False,
        hover=True,
        striped=True,
        size="sm",
        style={"fontSize": "13px"},
    )


# ── Layout ────────────────────────────────────────────────────────────────────
layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H5("MLOps & ROI", className="mb-0", style={"color": "#3E1C0A"}),
                    html.Small(
                        "Calcule o retorno financeiro dos modelos ajustando os parâmetros de negócio.",
                        style={"color": "#7A6555"},
                    ),
                ]),
                className="mb-3 mt-2",
            )
        ),

        dbc.Row(
            [
                # ── Painel de Inputs ──────────────────────────────────────────
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                "⚙️ Parâmetros de Negócio",
                                style={"backgroundColor": "#F7F4F0", "fontWeight": "600", "color": "#3E1C0A", "fontSize": "0.9rem"},
                            ),
                            dbc.CardBody(
                                [
                                    html.Label("Custo por unidade desperdiçada (R$)", style={"color": "#7A6555", "fontSize": "0.82rem"}),
                                    dcc.Input(
                                        id="roi-custo-desperdicio",
                                        type="number",
                                        value=5.0,
                                        min=0,
                                        step=0.5,
                                        debounce=True,
                                        className="mb-3",
                                        style={"width": "100%", "borderColor": "#D6C5B0"},
                                    ),

                                    html.Label("Margem por produto (0–1)", style={"color": "#7A6555", "fontSize": "0.82rem"}),
                                    dcc.Input(
                                        id="roi-margem",
                                        type="number",
                                        value=0.35,
                                        min=0,
                                        max=1,
                                        step=0.01,
                                        debounce=True,
                                        className="mb-3",
                                        style={"width": "100%", "borderColor": "#D6C5B0"},
                                    ),

                                    html.Label("Custo de campanha por cliente (R$)", style={"color": "#7A6555", "fontSize": "0.82rem"}),
                                    dcc.Input(
                                        id="roi-custo-campanha",
                                        type="number",
                                        value=10.0,
                                        min=0,
                                        step=1.0,
                                        debounce=True,
                                        className="mb-3",
                                        style={"width": "100%", "borderColor": "#D6C5B0"},
                                    ),

                                    html.Label("Custo de infraestrutura mensal (R$)", style={"color": "#7A6555", "fontSize": "0.82rem"}),
                                    dcc.Input(
                                        id="roi-custo-infra",
                                        type="number",
                                        value=500.0,
                                        min=0,
                                        step=50.0,
                                        debounce=True,
                                        className="mb-3",
                                        style={"width": "100%", "borderColor": "#D6C5B0"},
                                    ),

                                    html.Small(
                                        "Altere os valores e o ROI será recalculado automaticamente.",
                                        style={"color": "#7A6555"},
                                    ),
                                ]
                            ),
                        ],
                        style={"border": "1px solid #D6C5B0", "borderRadius": "6px"},
                    ),
                    md=3, className="mb-3",
                ),

                # ── KPIs de ROI + Gráfico ─────────────────────────────────────
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    kpi_card("ROI Forecasting", "—", icon="📦", color="#6B3A2A"),
                                    md=3, xs=6, className="mb-3",
                                    id="roi-kpi-forecasting-col",
                                ),
                                dbc.Col(
                                    kpi_card("ROI Churn", "—", icon="🔄", color="#A0664A"),
                                    md=3, xs=6, className="mb-3",
                                    id="roi-kpi-churn-col",
                                ),
                                dbc.Col(
                                    kpi_card("ROI Combinado", "—", icon="💼", color="#3E1C0A"),
                                    md=3, xs=6, className="mb-3",
                                    id="roi-kpi-combined-col",
                                ),
                                dbc.Col(
                                    kpi_card("Payback", "—", icon="📆", color="#7A6555"),
                                    md=3, xs=6, className="mb-3",
                                    id="roi-kpi-payback-col",
                                ),
                            ],
                            id="roi-kpi-row",
                        ),
                        dbc.Row(
                            dbc.Col(
                                dcc.Graph(
                                    id="roi-bar-chart",
                                    figure=roi_bar_chart(0, 0, 0, 0),
                                    config={"displayModeBar": False},
                                ),
                            )
                        ),
                    ],
                    md=9, className="mb-3",
                ),
            ]
        ),

        # ── Status dos Modelos ────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H6("Status dos Modelos", style={"color": "#3E1C0A", "marginBottom": "8px"}),
                        # Alertas de drift
                        dbc.Alert(
                            "⚠️ Data drift detectado nas features de entrada. Considere re-treino.",
                            color="warning",
                            is_open=_drift_detected,
                            dismissable=True,
                            style={"fontSize": "13px"},
                        ),
                        dbc.Alert(
                            "🔴 Concept drift detectado: degradação de MAPE > 20% em relação ao baseline.",
                            color="danger",
                            is_open=_concept_drift,
                            dismissable=True,
                            style={"fontSize": "13px"},
                        ),
                        _model_status_table(_model_status_data),
                    ],
                    md=6, className="mb-3",
                ),
                dbc.Col(
                    [
                        html.H6("Histórico de MAPE — Forecasting", style={"color": "#3E1C0A", "marginBottom": "8px"}),
                        dcc.Graph(
                            id="mlops-mape-history",
                            figure=mape_history_chart(_history),
                            config={"displayModeBar": False},
                        ),
                    ],
                    md=6, className="mb-3",
                ),
            ]
        ),
    ],
    fluid=True,
    className="py-2",
)
