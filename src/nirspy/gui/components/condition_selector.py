"""Condition selector component — checklist of experimental conditions."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def render_condition_selector(
    conditions: list[str] | None = None,
) -> html.Div:
    """Render a checklist of conditions for the HRF plot.

    The ``condition-selector`` element is **always** in the layout (even
    when no conditions are loaded) — its id is referenced as an ``Input``
    by the HRF callback, and Dash 4.x aborts the whole callback graph if
    that target is missing.

    Parameters
    ----------
    conditions:
        List of condition/event names.

    Returns
    -------
    html.Div
        Checklist (visible when conditions are present, hidden otherwise).
    """
    options = (
        [{"label": c, "value": c} for c in conditions] if conditions else []
    )
    values = list(conditions) if conditions else []
    visible = bool(conditions)

    return html.Div(
        [
            html.Label(
                "Conditions",
                className="fw-bold small mb-1",
                style={"display": "block" if visible else "none"},
            ),
            dbc.Checklist(
                id="condition-selector",
                options=options,
                value=values,
                inline=True,
                className="small",
                style={"display": "block" if visible else "none"},
            ),
            html.P(
                "No conditions available.",
                className="text-muted text-center py-2",
                style={"display": "none" if visible else "block"},
            ),
        ],
        id="condition-selector-content",
    )
