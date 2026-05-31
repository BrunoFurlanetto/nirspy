"""GLM runtime dialog for interactive pipeline execution (T-040).

Renders a ``dbc.Modal`` that intercepts execution immediately before a
``glm`` step in an interactive run.

The dialog allows the user to:

1. Review and edit the per-condition **stimulus duration** (seconds).
   Durations are pre-populated from ``raw.annotations``; the user can
   override them before the GLM design matrix is built.

2. Optionally define **condition groups**: named groups that merge several
   conditions into a single design-matrix regressor.  Each group has a
   user-supplied label and a multi-select list of conditions.

Footer
------
``[Cancel]``  ``[Run GLM]``

``"runtime-advance-btn"`` (on the Run button) is shared with the generic
runtime-advance callback so that ``advance_run`` fires and reads
``glm-runtime-state`` to build the ``params_override``.

``"runtime-cancel-btn"`` is also shared with the generic runtime dialog.

ID namespacing
--------------
Condition duration inputs use ``{"type": "glm-rt-duration", "cond": cond_name}``.
Group inputs use pattern-matching IDs with ``"group_idx"`` keys.

Stores
------
``"glm-runtime-state"`` — app-level store declared in ``layouts.py``.  Schema::

    {
        "available_conditions": list[str],
        "condition_durations": dict[str, float],
        "groups": [
            {
                "label": str,
                "conditions": list[str],
            },
            ...
        ],
    }
"""

from __future__ import annotations

import logging
from typing import Any

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html, no_update

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ID builders
# ---------------------------------------------------------------------------


def _duration_id(cond_name: str) -> dict[str, Any]:
    return {"type": "glm-rt-duration", "cond": cond_name}


def _group_label_id(group_idx: int) -> dict[str, Any]:
    return {"type": "glm-rt-group-label", "group_idx": group_idx}


def _group_conds_id(group_idx: int) -> dict[str, Any]:
    return {"type": "glm-rt-group-conds", "group_idx": group_idx}


def _group_remove_id(group_idx: int) -> dict[str, Any]:
    return {"type": "glm-rt-group-remove", "group_idx": group_idx}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_duration_row(cond: str, duration: float) -> dbc.Row:
    """Render a single condition-duration row."""
    return dbc.Row(
        [
            dbc.Col(
                html.Span(cond, className="small fw-semibold"),
                width=6,
                className="d-flex align-items-center",
            ),
            dbc.Col(
                dbc.Input(
                    id=_duration_id(cond),
                    type="number",
                    value=duration,
                    min=0.1,
                    step=0.1,
                    size="sm",
                    debounce=True,
                ),
                width=6,
            ),
        ],
        className="mb-1 g-2 align-items-center",
    )


def _render_group_card(
    group_idx: int,
    label: str,
    conditions: list[str],
    available_conditions: list[str],
) -> dbc.Card:
    """Render a single condition-group card."""
    options = [{"label": c, "value": c} for c in available_conditions]
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small("Group label", className="fw-bold d-block"),
                                dbc.Input(
                                    id=_group_label_id(group_idx),
                                    type="text",
                                    value=label,
                                    placeholder="e.g. Motor",
                                    size="sm",
                                    debounce=True,
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "- Remove",
                                id=_group_remove_id(group_idx),
                                color="danger",
                                outline=True,
                                size="sm",
                                className="w-100 mt-3",
                            ),
                            width=4,
                        ),
                    ],
                    className="mb-2 g-2",
                ),
                html.Small("Conditions", className="fw-bold d-block mb-1"),
                dcc.Dropdown(
                    id=_group_conds_id(group_idx),
                    options=options,
                    value=conditions,
                    multi=True,
                    placeholder="Select conditions for this group…",
                ),
            ],
            className="p-2",
        ),
        className="mb-2",
    )


def _render_groups_section(
    groups: list[dict[str, Any]],
    available_conditions: list[str],
) -> list[Any]:
    """Return rendered group cards."""
    return [
        _render_group_card(
            group_idx=idx,
            label=g.get("label", ""),
            conditions=list(g.get("conditions", [])),
            available_conditions=available_conditions,
        )
        for idx, g in enumerate(groups)
    ]


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------


def render_glm_runtime_dialog(
    block_spec: Any,
    current_idx: int,
    total: int,
    available_conditions: list[str],
    annotation_durations: dict[str, float],
) -> dbc.Modal:
    """Build the GLM runtime dialog Modal.

    Parameters
    ----------
    block_spec:
        The :class:`~nirspy.domain.block.BlockSpec` for the GLM block.
    current_idx:
        0-based index among enabled pipeline steps.
    total:
        Total number of enabled pipeline steps.
    available_conditions:
        Condition names extracted from upstream raw annotations.
        Empty list when no upstream Raw is available.
    annotation_durations:
        Per-condition durations pre-populated from raw annotations.
        Used as initial values for the editable duration inputs.

    Returns
    -------
    dbc.Modal
        Single-stage modal, initially open.
    """
    step_num = current_idx + 1
    header_text = f"GLM Configuration — Block {step_num}/{total}"

    # Build duration table
    if available_conditions:
        duration_header = dbc.Row(
            [
                dbc.Col(html.Small("Condition", className="fw-bold"), width=6),
                dbc.Col(html.Small("Duration (s)", className="fw-bold"), width=6),
            ],
            className="mb-1 g-2",
        )
        duration_rows: list[Any] = [
            _render_duration_row(
                cond,
                annotation_durations.get(cond, 1.0),
            )
            for cond in available_conditions
        ]
        duration_section: Any = html.Div(
            [
                html.H6("Condition Durations", className="mt-2 mb-2"),
                duration_header,
                *duration_rows,
            ]
        )
    else:
        duration_section = dbc.Alert(
            "No conditions detected upstream. "
            "Ensure a Beer-Lambert block precedes the GLM block and that "
            "the raw data has annotations.",
            color="warning",
            className="mb-2 py-2",
        )

    # Groups section
    groups_section = html.Div(
        [
            html.Hr(className="my-3"),
            html.H6("Condition Groups (optional)", className="mb-1"),
            html.Small(
                "Group conditions to merge them into a single regressor. "
                "Leave empty to model each condition independently.",
                className="text-muted d-block mb-2",
            ),
            html.Div([], id="glm-rt-groups-container"),
            dbc.Button(
                "+ Add Group",
                id="glm-rt-add-group-btn",
                color="primary",
                outline=True,
                size="sm",
                className="mt-1",
            ),
        ]
    )

    footer = dbc.ModalFooter(
        dbc.ButtonGroup(
            [
                dbc.Button(
                    "Cancel",
                    id="runtime-cancel-btn",
                    color="danger",
                    outline=True,
                    size="sm",
                ),
                dbc.Button(
                    "Run GLM",
                    id="runtime-advance-btn",
                    color="success",
                    size="sm",
                ),
            ],
            size="sm",
        ),
    )

    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(header_text),
                close_button=False,
            ),
            dbc.ModalBody(
                [
                    duration_section,
                    groups_section,
                ]
            ),
            footer,
        ],
        id="glm-runtime-modal",
        is_open=True,
        backdrop="static",
        keyboard=False,
        size="lg",
    )


# ---------------------------------------------------------------------------
# Public helper — build params_override from glm-runtime-state
# ---------------------------------------------------------------------------


def build_glm_params_override(
    glm_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert ``glm-runtime-state`` store data into a ``params_override`` dict.

    Parameters
    ----------
    glm_state:
        Current value of the ``"glm-runtime-state"`` Dash store.

    Returns
    -------
    dict or None
        Dict with ``condition_durations`` and/or ``per_condition_groups``
        when relevant data is present; ``None`` when ``glm_state`` is None
        or empty (block uses its own defaults).
    """
    if not glm_state:
        return None

    override: dict[str, Any] = {}

    # condition_durations: include only if the dict is non-empty
    raw_durations: dict[str, Any] = glm_state.get("condition_durations", {}) or {}
    condition_durations: dict[str, float] = {}
    for cond, val in raw_durations.items():
        try:
            condition_durations[cond] = float(val)
        except (TypeError, ValueError):
            continue
    if condition_durations:
        override["condition_durations"] = condition_durations

    # per_condition_groups: include only valid groups (label + at least 1 cond)
    groups: list[dict[str, Any]] = glm_state.get("groups", []) or []
    per_condition_groups: dict[str, list[str]] = {}
    for g in groups:
        label = (g.get("label") or "").strip()
        if not label:
            continue
        cond_names = [c for c in (g.get("conditions") or []) if c]
        if not cond_names:
            continue
        per_condition_groups[label] = cond_names
    if per_condition_groups:
        override["per_condition_groups"] = per_condition_groups

    return override if override else None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("glm-runtime-state", "data", allow_duplicate=True),
    Input({"type": "glm-rt-duration", "cond": ALL}, "value"),
    State("glm-runtime-state", "data"),
    prevent_initial_call=True,
)
def _glm_sync_durations(
    duration_values: list[float | None],
    glm_state: dict[str, Any] | None,
) -> Any:
    """Sync duration inputs back into ``glm-runtime-state``."""
    if not duration_values:
        return no_update

    from dash import callback_context  # local import to avoid circular issues

    triggered = callback_context.inputs_list[0] if callback_context.inputs_list else []
    if not triggered:
        return no_update

    state: dict[str, Any] = glm_state or {
        "available_conditions": [],
        "condition_durations": {},
        "groups": [],
    }
    new_durations: dict[str, float] = dict(state.get("condition_durations") or {})

    for item, val in zip(triggered, duration_values, strict=False):
        cond = item.get("id", {}).get("cond", "") if isinstance(item, dict) else ""
        if not cond:
            continue
        try:
            new_durations[cond] = float(val) if val is not None else 1.0
        except (TypeError, ValueError):
            new_durations[cond] = 1.0

    return {**state, "condition_durations": new_durations}


@callback(
    Output("glm-rt-groups-container", "children"),
    Output("glm-runtime-state", "data", allow_duplicate=True),
    Input("glm-rt-add-group-btn", "n_clicks"),
    State("glm-runtime-state", "data"),
    prevent_initial_call=True,
)
def _glm_add_group(
    n_clicks: int | None,
    glm_state: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Add a new empty group card."""
    if not n_clicks:
        return no_update, no_update

    state: dict[str, Any] = glm_state or {
        "available_conditions": [],
        "condition_durations": {},
        "groups": [],
    }
    groups: list[dict[str, Any]] = list(state.get("groups") or [])
    groups.append({"label": "", "conditions": []})
    new_state: dict[str, Any] = {**state, "groups": groups}

    available: list[str] = list(state.get("available_conditions") or [])
    cards = _render_groups_section(groups, available)
    return cards, new_state


@callback(
    Output("glm-rt-groups-container", "children", allow_duplicate=True),
    Output("glm-runtime-state", "data", allow_duplicate=True),
    Input({"type": "glm-rt-group-remove", "group_idx": ALL}, "n_clicks"),
    State("glm-runtime-state", "data"),
    prevent_initial_call=True,
)
def _glm_remove_group(
    n_clicks_list: list[int | None],
    glm_state: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Remove the group whose [- Remove] button was clicked."""
    if not any(n for n in n_clicks_list if n):
        return no_update, no_update

    clicked_idx: int | None = None
    for i, n in enumerate(n_clicks_list):
        if n:
            clicked_idx = i
            break

    if clicked_idx is None:
        return no_update, no_update

    state: dict[str, Any] = glm_state or {
        "available_conditions": [],
        "condition_durations": {},
        "groups": [],
    }
    groups: list[dict[str, Any]] = list(state.get("groups") or [])

    if 0 <= clicked_idx < len(groups):
        groups = [g for i, g in enumerate(groups) if i != clicked_idx]

    new_state: dict[str, Any] = {**state, "groups": groups}
    available: list[str] = list(state.get("available_conditions") or [])
    cards = _render_groups_section(groups, available)
    return cards, new_state


@callback(
    Output("glm-runtime-state", "data", allow_duplicate=True),
    Input({"type": "glm-rt-group-label", "group_idx": ALL}, "value"),
    Input({"type": "glm-rt-group-conds", "group_idx": ALL}, "value"),
    State("glm-runtime-state", "data"),
    prevent_initial_call=True,
)
def _glm_sync_groups(
    labels: list[str | None],
    conditions: list[list[str] | None],
    glm_state: dict[str, Any] | None,
) -> Any:
    """Sync group label and condition inputs back into ``glm-runtime-state``."""
    if not labels and not conditions:
        return no_update

    state: dict[str, Any] = glm_state or {
        "available_conditions": [],
        "condition_durations": {},
        "groups": [],
    }
    n_groups = max(len(labels), len(conditions))
    groups: list[dict[str, Any]] = []
    for i in range(n_groups):
        label = (labels[i] if i < len(labels) else None) or ""
        conds = list(
            (conditions[i] if i < len(conditions) else None) or []
        )
        groups.append({"label": label, "conditions": conds})

    return {**state, "groups": groups}
