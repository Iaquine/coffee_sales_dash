"""
kpi_card.py — Componente reutilizável de card de KPI.
"""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc


# Paleta de café
_COLORS = {
    "bg": "#FFFFFF",
    "surface": "#F7F4F0",
    "border": "#D6C5B0",
    "text": "#2C1A0E",
    "muted": "#7A6555",
    "accent": "#6B3A2A",
}


def kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = _COLORS["accent"],
    icon: str = "",
) -> dbc.Card:
    """
    Retorna um dbc.Card com título, valor destacado e subtítulo opcional.

    Parameters
    ----------
    title   : rótulo do KPI
    value   : valor formatado (ex: "R$ 1.234,56")
    subtitle: texto secundário abaixo do valor (ex: variação %)
    color   : cor da borda superior e do valor
    icon    : emoji/símbolo opcional exibido ao lado do título
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(
                    f"{icon} {title}".strip(),
                    className="mb-1",
                    style={"color": _COLORS["muted"], "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.05em"},
                ),
                html.H4(
                    value,
                    className="mb-0",
                    style={"color": color, "fontWeight": "700", "fontSize": "1.5rem"},
                ),
                html.Small(subtitle, style={"color": _COLORS["muted"]}) if subtitle else None,
            ]
        ),
        style={
            "borderTop": f"3px solid {color}",
            "borderRadius": "6px",
            "backgroundColor": _COLORS["surface"],
            "boxShadow": "none",
            "border": f"1px solid {_COLORS['border']}",
            "borderTopColor": color,
            "borderTopWidth": "3px",
        },
        className="h-100",
    )
