"""
charts.py — Funções reutilizáveis de gráficos Plotly para o dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# ── Paleta e layout base ──────────────────────────────────────────────────────
PALETTE = {
    "brown_dark": "#3E1C0A",
    "brown": "#6B3A2A",
    "brown_light": "#A0664A",
    "beige": "#D6C5B0",
    "cream": "#F7F4F0",
    "white": "#FFFFFF",
    "muted": "#7A6555",
    "blue": "#3b82d4",
    "green": "#2e7d32",
    "red": "#c62828",
    "orange": "#e65100",
}

SEGMENT_COLORS: dict[str, str] = {
    "Champions": PALETTE["brown_dark"],
    "Loyal": PALETTE["brown"],
    "Potential": PALETTE["brown_light"],
    "At Risk": PALETTE["orange"],
    "Lost": PALETTE["red"],
    "New": PALETTE["blue"],
}

_BASE_LAYOUT = dict(
    paper_bgcolor=PALETTE["white"],
    plot_bgcolor=PALETTE["cream"],
    font=dict(family="-apple-system, 'Segoe UI', system-ui, sans-serif", size=12, color=PALETTE["brown_dark"]),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["beige"], borderwidth=1),
    xaxis=dict(gridcolor=PALETTE["beige"], linecolor=PALETTE["beige"]),
    yaxis=dict(gridcolor=PALETTE["beige"], linecolor=PALETTE["beige"]),
)


def _base_fig(**kwargs) -> go.Figure:
    """Cria uma figura com o layout base aplicado."""
    fig = go.Figure()
    layout = {**_BASE_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# ── Aba 1 — Visão Geral ───────────────────────────────────────────────────────

def revenue_timeseries(df: pd.DataFrame) -> go.Figure:
    """
    Série temporal de receita diária + média móvel de 7 dias.
    Espera DataFrame com colunas: date, money.
    """
    daily = df.groupby("date")["money"].sum().reset_index().sort_values("date")
    daily["ma7"] = daily["money"].rolling(7).mean()

    fig = _base_fig(title="Receita Diária")
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["money"],
        mode="lines",
        name="Receita diária",
        line=dict(color=PALETTE["brown_light"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ma7"],
        mode="lines",
        name="Média móvel 7d",
        line=dict(color=PALETTE["brown_dark"], width=2, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Receita (R$)",
        hovermode="x unified",
    )
    return fig


def revenue_by_product(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Receita total por produto — barras horizontais, top N."""
    by_product = (
        df.groupby("coffee_name")["money"]
        .sum()
        .sort_values(ascending=True)
        .tail(top_n)
    )
    fig = _base_fig(title=f"Top {top_n} Produtos por Receita")
    fig.add_trace(go.Bar(
        x=by_product.values,
        y=by_product.index,
        orientation="h",
        marker_color=PALETTE["brown"],
        text=[f"R$ {v:,.0f}" for v in by_product.values],
        textposition="outside",
    ))
    fig.update_layout(xaxis_title="Receita (R$)", yaxis_title="", showlegend=False)
    return fig


def sales_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap de vendas por hora × dia da semana."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        df.groupby(["day_name", "hour"])["money"]
        .sum()
        .unstack(fill_value=0)
    )
    # Reindexar para garantir ordem dos dias
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig = _base_fig(title="Heatmap: Receita por Hora × Dia da Semana")
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}h" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, PALETTE["cream"]], [0.5, PALETTE["brown_light"]], [1, PALETTE["brown_dark"]]],
        hoverongaps=False,
        colorbar=dict(title="R$"),
    ))
    fig.update_layout(
        xaxis_title="Hora do dia",
        yaxis_title="Dia da semana",
        yaxis=dict(autorange="reversed", gridcolor=PALETTE["beige"]),
    )
    return fig


# ── Aba 2 — Demanda & Estoque ─────────────────────────────────────────────────

def forecast_chart(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, product: str) -> go.Figure:
    """
    Série histórica dos últimos 60 dias + previsões de 7 e 30 dias com banda de confiança.

    hist_df    : DataFrame com colunas date, money, coffee_name
    forecast_df: DataFrame com colunas product, date, predicted_revenue, lower_bound, upper_bound, horizon
    """
    # Histórico últimos 60 dias
    hist = (
        hist_df[hist_df["coffee_name"] == product]
        .groupby("date")["money"].sum()
        .reset_index()
        .sort_values("date")
    )
    if not hist.empty:
        cutoff = hist["date"].max() - pd.Timedelta(days=60)
        hist = hist[hist["date"] >= cutoff]

    fc = forecast_df[forecast_df["product"] == product].sort_values("date")
    fc7 = fc[fc["horizon"] == 7]
    fc30 = fc[fc["horizon"] == 30]

    fig = _base_fig(title=f"Previsão de Demanda — {product}")

    # Histórico
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["money"],
            mode="lines",
            name="Histórico (60d)",
            line=dict(color=PALETTE["brown_light"], width=2),
        ))

    def _add_forecast(fc_part: pd.DataFrame, horizon: int, color: str) -> None:
        if fc_part.empty:
            return
        # Banda de confiança
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_part["date"], fc_part["date"].iloc[::-1]]),
            y=pd.concat([fc_part["upper_bound"], fc_part["lower_bound"].iloc[::-1]]),
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ))
        # Linha central
        fig.add_trace(go.Scatter(
            x=fc_part["date"], y=fc_part["predicted_revenue"],
            mode="lines+markers",
            name=f"Previsão {horizon}d",
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=5),
        ))

    _add_forecast(fc7, 7, PALETTE["brown"])
    _add_forecast(fc30, 30, PALETTE["blue"])

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Receita Prevista (R$)",
        hovermode="x unified",
    )
    return fig


# ── Aba 3 — Clientes & Churn ──────────────────────────────────────────────────

def segment_scatter(segments_df: pd.DataFrame) -> go.Figure:
    """Scatter plot Recency × Frequency, tamanho = Monetary, cor = segment_name."""
    fig = _base_fig(title="Mapa de Segmentos: Recência × Frequência")
    for seg, grp in segments_df.groupby("segment_name"):
        color = SEGMENT_COLORS.get(str(seg), PALETTE["brown_light"])
        fig.add_trace(go.Scatter(
            x=grp["recency_days"],
            y=grp["frequency"],
            mode="markers",
            name=str(seg),
            marker=dict(
                size=grp["monetary"].clip(lower=1) ** 0.4,  # escala visual razoável
                color=color,
                opacity=0.75,
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Recência: %{x} dias<br>"
                "Frequência: %{y}<br>"
                "Monetário: R$ %{customdata[1]:.2f}<extra></extra>"
            ),
            customdata=list(zip(grp["card"], grp["monetary"])),
        ))
    fig.update_layout(xaxis_title="Recência (dias)", yaxis_title="Frequência (compras)")
    return fig


def segment_donut(segments_df: pd.DataFrame) -> go.Figure:
    """Pizza/donut com distribuição de segmentos por número de clientes."""
    counts = segments_df["segment_name"].value_counts()
    colors = [SEGMENT_COLORS.get(s, PALETTE["brown_light"]) for s in counts.index]

    fig = _base_fig(title="Distribuição de Segmentos")
    fig.add_trace(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} clientes (%{percent})<extra></extra>",
    ))
    return fig


# ── Aba 4 — MLOps & ROI ───────────────────────────────────────────────────────

def roi_bar_chart(
    forecast_benefit: float,
    forecast_cost: float,
    churn_benefit: float,
    churn_cost: float,
) -> go.Figure:
    """Gráfico de barras: benefício vs custo para Forecasting e Churn."""
    fig = _base_fig(title="Benefício vs. Custo dos Modelos")
    categories = ["Forecasting", "Churn"]
    benefits = [forecast_benefit, churn_benefit]
    costs = [forecast_cost, churn_cost]

    fig.add_trace(go.Bar(
        name="Benefício (R$)",
        x=categories,
        y=benefits,
        marker_color=PALETTE["green"],
        text=[f"R$ {v:,.0f}" for v in benefits],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Custo (R$)",
        x=categories,
        y=costs,
        marker_color=PALETTE["red"],
        text=[f"R$ {v:,.0f}" for v in costs],
        textposition="outside",
    ))
    fig.update_layout(barmode="group", yaxis_title="R$", xaxis_title="")
    return fig


def mape_history_chart(history: list[dict]) -> go.Figure:
    """Histórico de MAPE ao longo dos runs de forecasting."""
    if not history:
        fig = _base_fig(title="Histórico de MAPE — Forecasting")
        fig.add_annotation(text="Sem histórico disponível", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    names = [h["run_name"] for h in history]
    mapes = [h["mape"] for h in history]

    fig = _base_fig(title="Histórico de MAPE — Forecasting")
    fig.add_trace(go.Scatter(
        x=list(range(len(names))),
        y=mapes,
        mode="lines+markers",
        name="MAPE (%)",
        line=dict(color=PALETTE["brown"], width=2),
        marker=dict(size=8, color=PALETTE["brown_dark"]),
        text=names,
        hovertemplate="<b>%{text}</b><br>MAPE: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickvals=list(range(len(names))), ticktext=names, tickangle=-20),
        yaxis_title="MAPE (%)",
    )
    return fig
