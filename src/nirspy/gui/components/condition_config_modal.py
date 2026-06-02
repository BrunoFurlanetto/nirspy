"""Condition Configuration Modal (T-042i).

Allows users to configure global conditions after a SNIRF file is loaded.
The modal is opened automatically when the LoadSnirf block produces
annotations, and lets users:

1. Review/rename each detected condition.
2. Set per-condition temporal windows (tmin, tmax, baseline).
3. Select which occurrences to include (with collapse for > 50).
4. Optionally define named condition groups.

The confirmed configuration is serialised via
``nirspy.domain.conditions.global_conditions_to_dict`` and stored in
``global-conditions-store``.

ID convention
-------------
- ``"condition-config-modal"``          — the dbc.Modal
- ``"condition-config-conditions-container"``  — div with per-condition cards
- ``"condition-config-groups-container"``      — div with group cards
- ``"condition-config-apply-btn"``       — Apply / confirm button
- ``"condition-config-cancel-btn"``      — Cancel button
- ``"condition-config-warning"``         — warning div (hidden by default)
- ``"condition-config-state"``           — dcc.Store (declared in layouts.py)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html, no_update

logger = logging.getLogger(__name__)


def _coerce_float(value: float | None, default: float) -> float:
    """Return float(value) when value is not None, else default.

    Avoids the ``value or default`` pattern which treats ``0.0`` as falsy,
    incorrectly substituting the default when the user explicitly enters zero.
    """
    return float(value) if value is not None else default


# ---------------------------------------------------------------------------
# Helper: build conditions structure from raw annotations
# ---------------------------------------------------------------------------


def build_conditions_from_annotations(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect unique conditions and build initial modal structure.

    Parameters
    ----------
    annotations:
        List of annotation dicts, each with keys
        ``"description"``, ``"onset"`` (float, seconds), and
        ``"duration"`` (float, seconds).

    Returns
    -------
    list[dict]
        One entry per unique condition description::

            {
                "name": str,           # display/rename name (starts equal to original)
                "original_name": str,  # immutable key matching annotation description
                "duration": float,     # median duration across occurrences (> 0)
                "tmin": float,         # -2.0 default
                "tmax": float,         # 18.0 default
                "baseline_tmin": float,  # -2.0 default
                "baseline_tmax": float,  # 0.0 default
                "occurrences": [       # sorted by onset
                    {"idx": int, "onset": float, "selected": bool},
                    ...
                ],
            }
    """
    # Group annotations by description, skip BAD/boundary annotations
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_idx = 0
    for ann in sorted(annotations, key=lambda a: a.get("onset", 0.0)):
        desc: str = str(ann.get("description", ""))
        if desc.startswith("BAD") or desc in ("", "boundary"):
            global_idx += 1
            continue
        groups[desc].append(
            {
                "idx": global_idx,
                "onset": float(ann.get("onset", 0.0)),
                "duration": float(ann.get("duration", 1.0)),
            }
        )
        global_idx += 1

    conditions: list[dict[str, Any]] = []
    for desc, occurrences in groups.items():
        durations = [o["duration"] for o in occurrences if o["duration"] > 0]
        median_dur = float(sorted(durations)[len(durations) // 2]) if durations else 1.0

        occ_list = [
            {"idx": o["idx"], "onset": o["onset"], "selected": True}
            for o in occurrences
        ]
        conditions.append(
            {
                "name": desc,
                "original_name": desc,
                "duration": max(median_dur, 0.001),
                "tmin": -2.0,
                "tmax": 18.0,
                "baseline_tmin": -2.0,
                "baseline_tmax": 0.0,
                "occurrences": occ_list,
            }
        )

    return conditions


# ---------------------------------------------------------------------------
# Internal render helpers
# ---------------------------------------------------------------------------

_OCC_COLLAPSE_THRESHOLD = 50


def _occurrence_checklist(
    cond_idx: int,
    occurrences: list[dict[str, Any]],
) -> Any:
    """Render occurrence checklist, with collapse when > threshold."""
    items = [
        {
            "label": f"#{o['idx']}  onset {o['onset']:.2f}s",
            "value": o["idx"],
        }
        for o in occurrences
    ]
    selected = [o["idx"] for o in occurrences if o.get("selected", True)]

    checklist = dbc.Checklist(
        id={"type": "cond-cfg-occ", "cond_idx": cond_idx},
        options=items,
        value=selected,
        inline=False,
        className="small",
    )

    if len(occurrences) <= _OCC_COLLAPSE_THRESHOLD:
        return checklist

    # Large list — wrap in collapse with summary + select-all/deselect-all
    n_sel = len(selected)
    n_total = len(occurrences)
    summary_text = f"{n_sel} of {n_total} selected"
    collapse_id = f"occ-collapse-{cond_idx}"

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Small(summary_text, className="text-muted"),
                        width="auto",
                        className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Select All",
                                    id={
                                        "type": "cond-cfg-occ-all",
                                        "cond_idx": cond_idx,
                                        "action": "all",
                                    },
                                    size="sm",
                                    color="secondary",
                                    outline=True,
                                ),
                                dbc.Button(
                                    "Deselect All",
                                    id={
                                        "type": "cond-cfg-occ-all",
                                        "cond_idx": cond_idx,
                                        "action": "none",
                                    },
                                    size="sm",
                                    color="secondary",
                                    outline=True,
                                ),
                            ],
                            size="sm",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Show/Hide",
                            id=f"occ-toggle-{cond_idx}",
                            size="sm",
                            color="link",
                            className="p-0",
                        ),
                        width="auto",
                    ),
                ],
                className="g-2 mb-1",
            ),
            dbc.Collapse(
                checklist,
                id=collapse_id,
                is_open=False,
            ),
        ]
    )


def _render_condition_card(
    cond_idx: int,
    cond: dict[str, Any],
) -> dbc.Card:
    """Render a single condition configuration card."""
    occurrences: list[dict[str, Any]] = cond.get("occurrences", [])
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small("Display name", className="fw-bold d-block"),
                                dbc.Input(
                                    id={"type": "cond-cfg-name", "cond_idx": cond_idx},
                                    type="text",
                                    value=cond.get("name", ""),
                                    size="sm",
                                ),
                                # Hidden original_name for tracking identity
                                dcc.Input(
                                    id={
                                        "type": "cond-cfg-orig",
                                        "cond_idx": cond_idx,
                                    },
                                    type="hidden",
                                    value=cond.get("original_name", ""),
                                ),
                            ],
                            width=5,
                        ),
                        dbc.Col(
                            [
                                html.Small("Duration (s)", className="fw-bold d-block"),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-duration",
                                        "cond_idx": cond_idx,
                                    },
                                    type="number",
                                    value=cond.get("duration", 1.0),
                                    min=0.001,
                                    step=0.1,
                                    size="sm",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            html.Small(
                                f"{len(occurrences)} occurrence(s)",
                                className="text-muted d-block mt-3",
                            ),
                            width=4,
                        ),
                    ],
                    className="mb-2 g-2",
                ),
                # Temporal window row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small("tmin", className="fw-bold d-block"),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-tmin",
                                        "cond_idx": cond_idx,
                                    },
                                    type="number",
                                    value=cond.get("tmin", -2.0),
                                    step=0.5,
                                    size="sm",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Small("tmax", className="fw-bold d-block"),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-tmax",
                                        "cond_idx": cond_idx,
                                    },
                                    type="number",
                                    value=cond.get("tmax", 18.0),
                                    step=0.5,
                                    size="sm",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Small(
                                    "baseline tmin", className="fw-bold d-block"
                                ),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-btmin",
                                        "cond_idx": cond_idx,
                                    },
                                    type="number",
                                    value=cond.get("baseline_tmin", -2.0),
                                    step=0.5,
                                    size="sm",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Small(
                                    "baseline tmax", className="fw-bold d-block"
                                ),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-btmax",
                                        "cond_idx": cond_idx,
                                    },
                                    type="number",
                                    value=cond.get("baseline_tmax", 0.0),
                                    step=0.5,
                                    size="sm",
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="mb-2 g-2",
                ),
                # Occurrences checklist
                html.Small("Occurrences to include", className="fw-bold d-block mb-1"),
                _occurrence_checklist(cond_idx, occurrences),
            ],
            className="p-2",
        ),
        className="mb-3",
    )


def _render_conditions_section(
    conditions: list[dict[str, Any]],
) -> list[Any]:
    """Return list of condition cards."""
    return [
        _render_condition_card(idx, cond) for idx, cond in enumerate(conditions)
    ]


def _render_group_card(
    group_idx: int,
    group: dict[str, Any],
    available_condition_names: list[str],
) -> dbc.Card:
    """Render a single group configuration card."""
    options = [{"label": n, "value": n} for n in available_condition_names]
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small("Group label", className="fw-bold d-block"),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-grp-label",
                                        "group_idx": group_idx,
                                    },
                                    type="text",
                                    value=group.get("label", ""),
                                    placeholder="e.g. Motor",
                                    size="sm",
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "- Remove",
                                id={
                                    "type": "cond-cfg-grp-remove",
                                    "group_idx": group_idx,
                                },
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
                    id={
                        "type": "cond-cfg-grp-conds",
                        "group_idx": group_idx,
                    },
                    options=options,
                    value=list(group.get("conditions", [])),
                    multi=True,
                    placeholder="Select conditions for this group…",
                    className="mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small("tmin", className="fw-bold d-block"),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-grp-tmin",
                                        "group_idx": group_idx,
                                    },
                                    type="number",
                                    value=group.get("tmin", -2.0),
                                    step=0.5,
                                    size="sm",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Small("tmax", className="fw-bold d-block"),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-grp-tmax",
                                        "group_idx": group_idx,
                                    },
                                    type="number",
                                    value=group.get("tmax", 18.0),
                                    step=0.5,
                                    size="sm",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Small(
                                    "baseline tmin", className="fw-bold d-block"
                                ),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-grp-btmin",
                                        "group_idx": group_idx,
                                    },
                                    type="number",
                                    value=group.get("baseline_tmin", -2.0),
                                    step=0.5,
                                    size="sm",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Small(
                                    "baseline tmax", className="fw-bold d-block"
                                ),
                                dbc.Input(
                                    id={
                                        "type": "cond-cfg-grp-btmax",
                                        "group_idx": group_idx,
                                    },
                                    type="number",
                                    value=group.get("baseline_tmax", 0.0),
                                    step=0.5,
                                    size="sm",
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="g-2",
                ),
            ],
            className="p-2",
        ),
        className="mb-2",
    )


def _render_groups_section(
    groups: list[dict[str, Any]],
    available_condition_names: list[str],
) -> list[Any]:
    """Return list of group cards."""
    return [
        _render_group_card(idx, grp, available_condition_names)
        for idx, grp in enumerate(groups)
    ]


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------


def render_condition_config_modal() -> dbc.Modal:
    """Build and return the condition config Modal.

    The modal starts closed (``is_open=False``). It is opened by the
    ``open_condition_config_modal`` callback when a SNIRF file is loaded with
    annotations.

    Returns
    -------
    dbc.Modal
        The fully formed modal. Include this once in ``layouts.py``.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Configure Conditions"),
                close_button=False,
            ),
            dbc.ModalBody(
                [
                    # Warning area
                    html.Div(
                        id="condition-config-warning",
                        style={"display": "none"},
                    ),
                    # Section 1 — per-condition
                    html.H6("Conditions", className="mb-2"),
                    html.Small(
                        "Review and configure each condition detected in the SNIRF file. "
                        "You can rename conditions, adjust temporal windows, and select "
                        "which occurrences to include.",
                        className="text-muted d-block mb-3",
                    ),
                    html.Div(
                        id="condition-config-conditions-container",
                        children=[],
                    ),
                    html.Hr(className="my-3"),
                    # Section 2 — groups
                    html.H6("Condition Groups (optional)", className="mb-1"),
                    html.Small(
                        "Group conditions to analyse them together with shared "
                        "temporal windows. Leave empty to treat each condition "
                        "independently.",
                        className="text-muted d-block mb-2",
                    ),
                    html.Div(
                        id="condition-config-groups-container",
                        children=[],
                    ),
                    dbc.Button(
                        "+ Add Group",
                        id="condition-config-add-group-btn",
                        color="primary",
                        outline=True,
                        size="sm",
                        className="mt-1",
                    ),
                ],
            ),
            dbc.ModalFooter(
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "Cancel",
                            id="condition-config-cancel-btn",
                            color="danger",
                            outline=True,
                            size="sm",
                        ),
                        dbc.Button(
                            "Apply",
                            id="condition-config-apply-btn",
                            color="success",
                            size="sm",
                        ),
                    ],
                    size="sm",
                ),
            ),
        ],
        id="condition-config-modal",
        is_open=False,
        size="xl",
        scrollable=True,
        backdrop="static",
        keyboard=False,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("condition-config-conditions-container", "children"),
    Output("condition-config-groups-container", "children"),
    Output("condition-config-modal", "is_open"),
    Output("condition-config-state", "data"),
    Input("condition-config-state", "data"),
    prevent_initial_call=True,
)
def _populate_modal(
    state: dict[str, Any] | None,
) -> tuple[Any, Any, Any, Any]:
    """Re-render the modal content when condition-config-state changes externally.

    This callback is triggered when the state store is written from outside
    (e.g. after a SNIRF file is loaded). It renders the condition cards and
    group cards and opens the modal.
    """
    if not state:
        return no_update, no_update, no_update, no_update

    # Only open when explicitly requested
    if not state.get("_open", False):
        return no_update, no_update, no_update, no_update

    conditions: list[dict[str, Any]] = state.get("conditions", [])
    groups: list[dict[str, Any]] = state.get("groups", [])
    available_names = [c.get("name", "") for c in conditions]

    cond_cards = _render_conditions_section(conditions)
    group_cards = _render_groups_section(groups, available_names)

    # Clear the _open flag to prevent re-triggering
    new_state = {**state, "_open": False}
    return cond_cards, group_cards, True, new_state


@callback(
    Output("condition-config-groups-container", "children", allow_duplicate=True),
    Output("condition-config-state", "data", allow_duplicate=True),
    Input("condition-config-add-group-btn", "n_clicks"),
    State("condition-config-state", "data"),
    prevent_initial_call=True,
)
def _add_group(
    n_clicks: int | None,
    state: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Add an empty group card."""
    if not n_clicks:
        return no_update, no_update

    s: dict[str, Any] = state or {"conditions": [], "groups": []}
    groups: list[dict[str, Any]] = list(s.get("groups", []))
    groups.append(
        {
            "label": "",
            "conditions": [],
            "tmin": -2.0,
            "tmax": 18.0,
            "baseline_tmin": -2.0,
            "baseline_tmax": 0.0,
        }
    )
    new_state = {**s, "groups": groups}
    available_names = [c.get("name", "") for c in s.get("conditions", [])]
    cards = _render_groups_section(groups, available_names)
    return cards, new_state


@callback(
    Output("condition-config-groups-container", "children", allow_duplicate=True),
    Output("condition-config-state", "data", allow_duplicate=True),
    Input({"type": "cond-cfg-grp-remove", "group_idx": ALL}, "n_clicks"),
    State("condition-config-state", "data"),
    prevent_initial_call=True,
)
def _remove_group(
    n_clicks_list: list[int | None],
    state: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Remove the group whose Remove button was clicked."""
    if not any(n for n in n_clicks_list if n):
        return no_update, no_update

    clicked_idx: int | None = None
    for i, n in enumerate(n_clicks_list):
        if n:
            clicked_idx = i
            break

    if clicked_idx is None:
        return no_update, no_update

    s: dict[str, Any] = state or {"conditions": [], "groups": []}
    groups = [g for i, g in enumerate(s.get("groups", [])) if i != clicked_idx]
    new_state = {**s, "groups": groups}
    available_names = [c.get("name", "") for c in s.get("conditions", [])]
    cards = _render_groups_section(groups, available_names)
    return cards, new_state


@callback(
    Output("condition-config-state", "data", allow_duplicate=True),
    Input({"type": "cond-cfg-name", "cond_idx": ALL}, "value"),
    Input({"type": "cond-cfg-duration", "cond_idx": ALL}, "value"),
    Input({"type": "cond-cfg-tmin", "cond_idx": ALL}, "value"),
    Input({"type": "cond-cfg-tmax", "cond_idx": ALL}, "value"),
    Input({"type": "cond-cfg-btmin", "cond_idx": ALL}, "value"),
    Input({"type": "cond-cfg-btmax", "cond_idx": ALL}, "value"),
    Input({"type": "cond-cfg-occ", "cond_idx": ALL}, "value"),
    State("condition-config-state", "data"),
    State({"type": "cond-cfg-orig", "cond_idx": ALL}, "value"),
    prevent_initial_call=True,
)
def _sync_condition_inputs(  # noqa: PLR0913
    names: list[str | None],
    durations: list[float | None],
    tmins: list[float | None],
    tmaxs: list[float | None],
    btmins: list[float | None],
    btmaxs: list[float | None],
    occ_values: list[list[int] | None],
    state: dict[str, Any] | None,
    orig_names: list[str | None],
) -> Any:
    """Sync all per-condition inputs back into condition-config-state."""
    s: dict[str, Any] = state or {"conditions": [], "groups": []}
    existing: list[dict[str, Any]] = list(s.get("conditions", []))

    n = max(len(names), len(existing))
    updated: list[dict[str, Any]] = []
    for i in range(n):
        base = existing[i] if i < len(existing) else {}
        orig = (
            (orig_names[i] if i < len(orig_names) else None)
            or base.get("original_name", "")
        )
        name = (names[i] if i < len(names) else None) or base.get("name", orig)
        dur = (durations[i] if i < len(durations) else None)
        if dur is None:
            dur = base.get("duration", 1.0)
        tmin = (tmins[i] if i < len(tmins) else None)
        if tmin is None:
            tmin = base.get("tmin", -2.0)
        tmax = (tmaxs[i] if i < len(tmaxs) else None)
        if tmax is None:
            tmax = base.get("tmax", 18.0)
        btmin = (btmins[i] if i < len(btmins) else None)
        if btmin is None:
            btmin = base.get("baseline_tmin", -2.0)
        btmax = (btmaxs[i] if i < len(btmaxs) else None)
        if btmax is None:
            btmax = base.get("baseline_tmax", 0.0)

        # Merge occurrence selection back
        base_occs: list[dict[str, Any]] = base.get("occurrences", [])
        selected_set = set(occ_values[i] or []) if i < len(occ_values) else {
            o["idx"] for o in base_occs
        }
        new_occs = [
            {**o, "selected": o["idx"] in selected_set} for o in base_occs
        ]

        updated.append(
            {
                "name": name or orig,
                "original_name": orig,
                "duration": float(dur) if dur is not None else 1.0,
                "tmin": float(tmin),
                "tmax": float(tmax),
                "baseline_tmin": float(btmin),
                "baseline_tmax": float(btmax),
                "occurrences": new_occs,
            }
        )

    return {**s, "conditions": updated}


@callback(
    Output("condition-config-state", "data", allow_duplicate=True),
    Input({"type": "cond-cfg-grp-label", "group_idx": ALL}, "value"),
    Input({"type": "cond-cfg-grp-conds", "group_idx": ALL}, "value"),
    Input({"type": "cond-cfg-grp-tmin", "group_idx": ALL}, "value"),
    Input({"type": "cond-cfg-grp-tmax", "group_idx": ALL}, "value"),
    Input({"type": "cond-cfg-grp-btmin", "group_idx": ALL}, "value"),
    Input({"type": "cond-cfg-grp-btmax", "group_idx": ALL}, "value"),
    State("condition-config-state", "data"),
    prevent_initial_call=True,
)
def _sync_group_inputs(
    labels: list[str | None],
    conditions: list[list[str] | None],
    tmins: list[float | None],
    tmaxs: list[float | None],
    btmins: list[float | None],
    btmaxs: list[float | None],
    state: dict[str, Any] | None,
) -> Any:
    """Sync group inputs back into condition-config-state."""
    s: dict[str, Any] = state or {"conditions": [], "groups": []}
    n = max(len(labels), len(s.get("groups", [])))
    groups: list[dict[str, Any]] = []
    for i in range(n):
        groups.append(
            {
                "label": (labels[i] if i < len(labels) else None) or "",
                "conditions": list(
                    (conditions[i] if i < len(conditions) else None) or []
                ),
                "tmin": _coerce_float(
                    tmins[i] if i < len(tmins) else None, -2.0
                ),
                "tmax": _coerce_float(
                    tmaxs[i] if i < len(tmaxs) else None, 18.0
                ),
                "baseline_tmin": _coerce_float(
                    btmins[i] if i < len(btmins) else None, -2.0
                ),
                "baseline_tmax": _coerce_float(
                    btmaxs[i] if i < len(btmaxs) else None, 0.0
                ),
            }
        )
    return {**s, "groups": groups}
