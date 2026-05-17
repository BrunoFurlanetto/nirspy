"""Condition selector component — checklist of experimental conditions."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def render_condition_selector(
    conditions: list[str] | None = None,
) -> html.Div:
    """Render a checklist of conditions for the HRF plot.

    All conditions are checked by default.

    Parameters
    ----------
    conditions:
        List of condition/event names.

    Returns
    -------
    html.Div
        Checklist or placeholder.
    """
    if not conditions:
        return html.Div(
            html.P(
                "No conditions available.",
                className="text-muted text-center py-2",
            ),
            id="condition-selector-content",
        )

    options = [{"label": c, "value": c} for c in conditions]

    return html.Div(
        [
            html.Label(
                "Conditions",
                className="fw-bold small mb-1",
            ),
            dbc.Checklist(
                id="condition-selector",
                options=options,
                value=conditions,
                inline=True,
                className="small",
            ),
        ],
        id="condition-selector-content",
    )
