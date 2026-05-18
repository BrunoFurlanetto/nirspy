"""Per-condition temporal window editor for BlockAverage.

Renders a Switch (on/off) plus a dynamic table of condition rows,
each with a dropdown for condition name and four numeric inputs
(tmin, tmax, baseline_tmin, baseline_tmax).

This component is only used when the selected block is ``block_average``
and the field is ``per_condition_windows``.  The param_editor delegates
to :func:`render_condition_windows_editor` via a surgical check.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html


def _row_id(instance_id: str, condition: str, field: str) -> dict[str, str]:
    """Build a pattern-matching ID for a single cell."""
    return {
        "type": "cond-window-row",
        "instance_id": instance_id,
        "condition": condition,
        "field": field,
    }


def _render_row(
    instance_id: str,
    condition: str,
    values: dict[str, float],
    available_conditions: list[str] | None,
) -> dbc.Row:
    """Render one condition row (dropdown + 4 numeric inputs + remove btn)."""
    if available_conditions:
        cond_ctrl: Any = dbc.Select(
            id=_row_id(instance_id, condition, "condition_name"),
            options=[{"label": c, "value": c} for c in available_conditions],
            value=condition,
            size="sm",
        )
    else:
        cond_ctrl = dbc.Input(
            id=_row_id(instance_id, condition, "condition_name"),
            type="text",
            value=condition,
            size="sm",
            placeholder="Type condition name",
        )

    fields = ["tmin", "tmax", "baseline_tmin", "baseline_tmax"]
    inputs = []
    for f in fields:
        inputs.append(
            dbc.Col(
                dbc.Input(
                    id=_row_id(instance_id, condition, f),
                    type="number",
                    value=values.get(f, 0),
                    size="sm",
                    step=1.0,
                ),
                width=2,
            )
        )

    remove_btn = dbc.Col(
        dbc.Button(
            html.I(className="bi bi-x"),
            id={
                "type": "cond-window-remove",
                "instance_id": instance_id,
                "condition": condition,
            },
            color="danger",
            size="sm",
            outline=True,
        ),
        width=1,
        className="d-flex align-items-center",
    )

    return dbc.Row(
        [dbc.Col(cond_ctrl, width=3)] + inputs + [remove_btn],
        className="mb-1 g-1",
    )


def render_condition_windows_editor(
    instance_id: str,
    current_value: dict[str, Any] | None,
    available_conditions: list[str] | None = None,
) -> html.Div:
    """Render the per-condition windows editor widget.

    Parameters
    ----------
    instance_id:
        Unique pipeline step ID.
    current_value:
        Current ``per_condition_windows`` dict (may contain raw dicts
        or ConditionWindow-like objects).
    available_conditions:
        Condition names from the last pipeline run, or ``None``.

    Returns
    -------
    html.Div
        Complete widget with switch, header, rows and add button.
    """
    current_value = current_value or {}
    is_enabled = len(current_value) > 0

    switch = dbc.Switch(
        id={"type": "cond-window-switch", "instance_id": instance_id},
        label="Use per-condition windows",
        value=is_enabled,
        className="mb-2",
    )

    header = dbc.Row(
        [
            dbc.Col(html.Small("Condition", className="fw-bold"), width=3),
            dbc.Col(html.Small("tmin", className="fw-bold"), width=2),
            dbc.Col(html.Small("tmax", className="fw-bold"), width=2),
            dbc.Col(html.Small("bl_tmin", className="fw-bold"), width=2),
            dbc.Col(html.Small("bl_tmax", className="fw-bold"), width=2),
            dbc.Col(html.Small(""), width=1),
        ],
        className="mb-1 g-1",
    )

    rows = []
    for cond, win in current_value.items():
        if hasattr(win, "tmin"):
            vals = {
                "tmin": win.tmin,
                "tmax": win.tmax,
                "baseline_tmin": win.baseline_tmin,
                "baseline_tmax": win.baseline_tmax,
            }
        elif isinstance(win, dict):
            vals = win
        else:
            vals = {}
        rows.append(
            _render_row(instance_id, cond, vals, available_conditions)
        )

    add_btn = dbc.Button(
        "+ Add condition",
        id={"type": "cond-window-add", "instance_id": instance_id},
        color="secondary",
        size="sm",
        outline=True,
        className="mt-1",
    )

    hint = ""
    if not available_conditions:
        hint = "Run pipeline first to populate condition dropdown."

    table_div = html.Div(
        [header] + rows + [add_btn],
        id={"type": "cond-window-table", "instance_id": instance_id},
        style={"display": "block" if is_enabled else "none"},
    )

    children = [switch]
    if hint:
        children.append(
            html.Small(hint, className="text-muted d-block mb-1")
        )
    children.append(table_div)

    return html.Div(
        [
            html.Label(
                "Per-condition windows",
                className="small fw-bold mb-1",
            ),
        ]
        + children,
        className="mb-3 p-2 border rounded",
    )
