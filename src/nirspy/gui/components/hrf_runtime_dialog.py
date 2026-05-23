"""HRF-specialized runtime dialog for interactive pipeline execution (T-028).

Renders a two-stage wizard ``dbc.Modal`` that intercepts execution immediately
before a ``block_average`` step in an interactive run.

Stage 1 — Group conditions
--------------------------
Displays a list of condition-group cards (initially empty). Each card has:

- A text input for the group **label**.
- A multi-select ``dcc.Dropdown`` auto-populated with conditions from the
  upstream SNIRF file (via ``read_snirf_condition_names``).
- Four numeric inputs: ``tmin``, ``tmax``, ``baseline_tmin``,
  ``baseline_tmax``.
- A ``[- Remove]`` button.

``[+ Add group]`` appends a new empty card.
An orphan-conditions hint appears in grey when some conditions are unassigned.

Stage 2 — Timings per group
----------------------------
One row per group: read-only label + four editable numeric inputs
(tmin, tmax, baseline_tmin, baseline_tmax). Changes here update the
``hrf-runtime-state`` store via a separate pattern-matching callback so they
persist if the user goes back to Stage 1 and forward again.

Footer
------
Stage 1: ``[Cancel run]``  ``[Next: configure times →]``
Stage 2: ``[← Back to Stage 1]``  ``[Run BlockAverage with these groups]``

The "Run BlockAverage" button carries ``id="runtime-advance-btn"`` so that the
existing ``advance_run`` callback in ``runtime_callbacks.py`` fires and reads
``hrf-runtime-state`` to build the ``params_override``.

``"runtime-cancel-btn"`` is also shared with the generic runtime dialog.

ID namespacing
--------------
Stage 1 group inputs use ``type = "hrf-rt-time"`` so they can be read by a
pattern-matching ``ALL`` callback.  Stage 2 uses ``type = "hrf-rt-time2"`` to
avoid Dash complaining about duplicate IDs when both stages are in the DOM
simultaneously (Stage 2 container is ``display:none`` but still in the DOM).

Stores
------
``"hrf-runtime-state"`` — app-level store declared in ``layouts.py``.  Schema::

    {
        "groups": [
            {
                "label": str,
                "condition_names": list[str],
                "tmin": float,
                "tmax": float,
                "baseline_tmin": float,
                "baseline_tmax": float,
            },
            ...
        ],
        "available_conditions": list[str] | None,
    }

``"hrf-runtime-stage"`` — transient store embedded in the modal children;
    tracks current stage (1 or 2).
"""

from __future__ import annotations

import logging
from typing import Any

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html, no_update

from nirspy.gui.components.param_metadata import metadata_for

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGE_STORE_ID = "hrf-runtime-stage"

_TIME_FIELDS: list[tuple[str, str]] = [
    ("tmin", "tmin"),
    ("tmax", "tmax"),
    ("baseline_tmin", "bl_tmin"),
    ("baseline_tmax", "bl_tmax"),
]

# ---------------------------------------------------------------------------
# ID builders
# ---------------------------------------------------------------------------


def _label_id(group_idx: int) -> dict[str, Any]:
    return {"type": "hrf-rt-label", "group_idx": group_idx}


def _cond_id(group_idx: int) -> dict[str, Any]:
    return {"type": "hrf-rt-cond", "group_idx": group_idx}


def _time_id(group_idx: int, field: str) -> dict[str, Any]:
    """Stage 1 group time input ID."""
    return {"type": "hrf-rt-time", "group_idx": group_idx, "field": field}


def _time2_id(group_idx: int, field: str) -> dict[str, Any]:
    """Stage 2 group time input ID (distinct type to avoid duplicate-ID conflicts)."""
    return {"type": "hrf-rt-time2", "group_idx": group_idx, "field": field}


def _remove_id(group_idx: int) -> dict[str, Any]:
    return {"type": "hrf-rt-remove", "group_idx": group_idx}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _numeric_attrs_for(field_name: str) -> dict[str, Any]:
    """Return HTML5 min/max/step attrs from ParamMeta for ``block_average``."""
    meta = metadata_for("block_average", field_name)
    attrs: dict[str, Any] = {}
    if meta is None:
        return attrs
    if meta.min is not None:
        attrs["min"] = meta.min
    if meta.max is not None:
        attrs["max"] = meta.max
    if meta.step is not None:
        attrs["step"] = meta.step
    return attrs


def _time_defaults_from_params(global_params: dict[str, Any]) -> dict[str, float]:
    """Return tmin/tmax/baseline defaults from a BlockAverageParams dict."""
    return {
        "tmin": float(global_params.get("tmin") or -2.0),
        "tmax": float(global_params.get("tmax") or 18.0),
        "baseline_tmin": float(global_params.get("baseline_tmin") or -2.0),
        "baseline_tmax": float(global_params.get("baseline_tmax") or 0.0),
    }


def _collect_orphans(
    available_conditions: list[str] | None,
    groups: list[dict[str, Any]],
) -> list[str]:
    """Return conditions not assigned to any group."""
    if not available_conditions:
        return []
    assigned: set[str] = set()
    for g in groups:
        for c in g.get("condition_names", []):
            assigned.add(c)
    return [c for c in available_conditions if c not in assigned]


def _collect_duplicates(groups: list[dict[str, Any]]) -> set[str]:
    """Return conditions that appear in more than one group."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for g in groups:
        for c in g.get("condition_names", []):
            if c in seen:
                dupes.add(c)
            seen.add(c)
    return dupes


# ---------------------------------------------------------------------------
# Stage 1 — group card renderer
# ---------------------------------------------------------------------------


def _render_group_card(
    group_idx: int,
    label: str,
    condition_names: list[str],
    time_vals: dict[str, float],
    available_conditions: list[str] | None,
    duplicated_conditions: set[str],
) -> dbc.Card:
    """Render a single HRF-runtime condition-group card (Stage 1)."""
    options = (
        [{"label": c, "value": c} for c in available_conditions]
        if available_conditions
        else []
    )
    invalid = [c for c in condition_names if c in duplicated_conditions]
    dropdown_style: dict[str, Any] = (
        {"border": "1px solid var(--bs-danger)"} if invalid else {}
    )

    time_inputs: list[Any] = []
    for field_name, label_short in _TIME_FIELDS:
        attrs = _numeric_attrs_for(field_name)
        if "step" not in attrs:
            attrs["step"] = 0.1
        time_inputs.append(
            dbc.Col(
                [
                    html.Small(label_short, className="fw-bold d-block"),
                    dbc.Input(
                        id=_time_id(group_idx, field_name),
                        type="number",
                        value=time_vals.get(field_name, 0.0),
                        size="sm",
                        debounce=True,
                        **attrs,
                    ),
                ],
                width=3,
            )
        )

    cond_hint: list[Any] = []
    if not available_conditions:
        cond_hint = [
            html.Small(
                "Set the LoadSnirf path first to get condition names.",
                className="text-muted d-block mb-1",
            )
        ]

    conflict_msg: list[Any] = []
    if invalid:
        conflict_msg = [
            html.Small(
                f"Conflict: {', '.join(sorted(invalid))} already in another group.",
                className="text-danger d-block mt-1",
            )
        ]

    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small("Group label", className="fw-bold d-block"),
                                dbc.Input(
                                    id=_label_id(group_idx),
                                    type="text",
                                    value=label,
                                    placeholder="e.g. Long stimuli",
                                    size="sm",
                                    debounce=True,
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "- Remove",
                                id=_remove_id(group_idx),
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
                *cond_hint,
                dcc.Dropdown(
                    id=_cond_id(group_idx),
                    options=options,
                    value=condition_names,
                    multi=True,
                    placeholder="Select conditions for this group…",
                    style=dropdown_style,
                ),
                *conflict_msg,
                dbc.Row(time_inputs, className="mt-2 g-1"),
            ],
            className="p-2",
        ),
        className="mb-2",
    )


# ---------------------------------------------------------------------------
# Stage renderers
# ---------------------------------------------------------------------------


def _render_groups_list(
    groups: list[dict[str, Any]],
    available_conditions: list[str] | None,
) -> list[Any]:
    """Return list of group card elements for Stage 1."""
    duplicated = _collect_duplicates(groups)
    return [
        _render_group_card(
            group_idx=idx,
            label=g.get("label", ""),
            condition_names=list(g.get("condition_names", [])),
            time_vals={
                "tmin": g.get("tmin", -2.0),
                "tmax": g.get("tmax", 18.0),
                "baseline_tmin": g.get("baseline_tmin", -2.0),
                "baseline_tmax": g.get("baseline_tmax", 0.0),
            },
            available_conditions=available_conditions,
            duplicated_conditions=duplicated,
        )
        for idx, g in enumerate(groups)
    ]


def _render_stage1_body(
    groups: list[dict[str, Any]],
    available_conditions: list[str] | None,
) -> html.Div:
    """Full Stage 1 body: condition-group cards + add button + orphan warning."""
    cards = _render_groups_list(groups, available_conditions)

    no_groups_hint: list[Any] = (
        [
            html.Small(
                "No groups defined. Click [+ Add group] to create one.",
                className="text-muted d-block mb-2",
            )
        ]
        if not groups
        else []
    )

    cond_hint: list[Any] = (
        [
            dbc.Alert(
                "No SNIRF conditions loaded — conditions will not be "
                "auto-populated. Ensure a LoadSnirf block precedes this one.",
                color="warning",
                className="mb-2 py-2",
            )
        ]
        if not available_conditions
        else []
    )

    orphans = _collect_orphans(available_conditions, groups)
    orphan_warn: list[Any] = (
        [
            dbc.Alert(
                html.Small(
                    f"Unassigned conditions: {', '.join(sorted(orphans))}",
                    className="text-secondary",
                ),
                color="light",
                className="mt-2 mb-0 py-1",
            )
        ]
        if orphans
        else []
    )

    return html.Div(
        [
            *cond_hint,
            *no_groups_hint,
            html.Div(cards, id="hrf-rt-cards-container"),
            dbc.Button(
                "+ Add group",
                id="hrf-rt-add-btn",
                color="primary",
                outline=True,
                size="sm",
                className="mb-2 mt-1",
            ),
            *orphan_warn,
        ],
    )


def _render_stage2_body(groups: list[dict[str, Any]]) -> html.Div:
    """Full Stage 2 body: timings table with ``hrf-rt-time2`` IDs."""
    if not groups:
        return html.Div(
            dbc.Alert(
                "No groups defined. Go back to Stage 1 and add at least one group.",
                color="warning",
            )
        )

    header = dbc.Row(
        [
            dbc.Col(html.Small("Group label", className="fw-bold"), width=4),
            *[
                dbc.Col(html.Small(short, className="fw-bold"), width=2)
                for _, short in _TIME_FIELDS
            ],
        ],
        className="mb-1 g-1",
    )

    rows: list[Any] = []
    for idx, g in enumerate(groups):
        label = g.get("label") or f"Group {idx + 1}"
        label_cell = dbc.Col(
            html.Div(
                html.Span(label, className="fw-semibold small"),
                className="d-flex align-items-center h-100",
            ),
            width=4,
        )
        time_cells: list[Any] = []
        for field_name, _short in _TIME_FIELDS:
            attrs = _numeric_attrs_for(field_name)
            if "step" not in attrs:
                attrs["step"] = 0.1
            time_cells.append(
                dbc.Col(
                    dbc.Input(
                        id=_time2_id(idx, field_name),
                        type="number",
                        value=g.get(field_name, 0.0),
                        size="sm",
                        debounce=True,
                        **attrs,
                    ),
                    width=2,
                )
            )
        rows.append(
            dbc.Row(
                [label_cell, *time_cells],
                className="mb-1 g-1 align-items-center",
            )
        )

    return html.Div(
        [
            html.Small(
                f"{len(groups)} group(s) — edit epoch windows then click Run.",
                className="text-muted d-block mb-2",
            ),
            header,
            *rows,
        ],
    )


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------


def render_hrf_runtime_dialog(
    block_spec: Any,
    current_idx: int,
    total: int,
    available_conditions: list[str] | None = None,
    current_params: dict[str, Any] | None = None,
) -> dbc.Modal:
    """Build the HRF-specialized runtime wizard Modal for a ``block_average`` step.

    Parameters
    ----------
    block_spec:
        The :class:`~nirspy.domain.block.BlockSpec` for the BlockAverage block.
    current_idx:
        0-based index among enabled pipeline steps.
    total:
        Total number of enabled pipeline steps.
    available_conditions:
        Condition names from the upstream SNIRF file.  ``None`` when no SNIRF
        file is reachable yet.
    current_params:
        Current ``BlockAverageParams`` field values for seeding time defaults.

    Returns
    -------
    dbc.Modal
        Two-stage wizard, initially open on Stage 1.
    """
    step_num = current_idx + 1
    header_text = (
        f"Block {step_num}/{total}: {block_spec.display_name} — HRF Groups"
    )

    defaults = _time_defaults_from_params(current_params or {})
    initial_groups: list[dict[str, Any]] = []

    stage1_body = _render_stage1_body(initial_groups, available_conditions)
    stage2_body = _render_stage2_body(initial_groups)

    footer_stage1 = dbc.ModalFooter(
        dbc.ButtonGroup(
            [
                dbc.Button(
                    "Cancel run",
                    id="runtime-cancel-btn",
                    color="danger",
                    outline=True,
                    size="sm",
                ),
                dbc.Button(
                    "Next: configure times →",
                    id="hrf-rt-next-btn",
                    color="primary",
                    size="sm",
                ),
            ],
            size="sm",
        ),
        id="hrf-rt-footer-stage1",
    )

    footer_stage2 = dbc.ModalFooter(
        dbc.ButtonGroup(
            [
                dbc.Button(
                    "← Back to Stage 1",
                    id="hrf-rt-back-btn",
                    color="secondary",
                    outline=True,
                    size="sm",
                ),
                dbc.Button(
                    "Run BlockAverage with these groups",
                    id="runtime-advance-btn",
                    color="success",
                    size="sm",
                ),
            ],
            size="sm",
        ),
        id="hrf-rt-footer-stage2",
        style={"display": "none"},
    )

    return dbc.Modal(
        [
            dcc.Store(id=_STAGE_STORE_ID, data=1),
            dcc.Store(id="hrf-rt-defaults", data=defaults),
            dbc.ModalHeader(
                dbc.ModalTitle(header_text),
                close_button=False,
            ),
            dbc.ModalBody(
                [
                    html.Div(
                        stage1_body,
                        id="hrf-rt-stage1-container",
                        style={"display": "block"},
                    ),
                    html.Div(
                        stage2_body,
                        id="hrf-rt-stage2-container",
                        style={"display": "none"},
                    ),
                ]
            ),
            footer_stage1,
            footer_stage2,
        ],
        id="hrf-runtime-modal",
        is_open=True,
        backdrop="static",
        keyboard=False,
        size="xl",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("hrf-rt-cards-container", "children"),
    Output("hrf-runtime-state", "data", allow_duplicate=True),
    Input("hrf-rt-add-btn", "n_clicks"),
    State("hrf-runtime-state", "data"),
    State("hrf-rt-defaults", "data"),
    prevent_initial_call=True,
)
def _hrf_add_group(
    n_clicks: int | None,
    hrf_state: dict[str, Any] | None,
    defaults: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Add a new empty group card to Stage 1."""
    if not n_clicks:
        return no_update, no_update

    state = hrf_state or {"groups": [], "available_conditions": None}
    groups: list[dict[str, Any]] = list(state.get("groups", []))
    d = defaults or {}
    groups.append(
        {
            "label": "",
            "condition_names": [],
            "tmin": float(d.get("tmin", -2.0)),
            "tmax": float(d.get("tmax", 18.0)),
            "baseline_tmin": float(d.get("baseline_tmin", -2.0)),
            "baseline_tmax": float(d.get("baseline_tmax", 0.0)),
        }
    )
    new_state: dict[str, Any] = {**state, "groups": groups}
    available = state.get("available_conditions")
    cards = _render_groups_list(groups, available)
    return cards, new_state


@callback(
    Output("hrf-rt-cards-container", "children", allow_duplicate=True),
    Output("hrf-runtime-state", "data", allow_duplicate=True),
    Input({"type": "hrf-rt-remove", "group_idx": ALL}, "n_clicks"),
    State("hrf-runtime-state", "data"),
    prevent_initial_call=True,
)
def _hrf_remove_group(
    n_clicks_list: list[int | None],
    hrf_state: dict[str, Any] | None,
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

    state = hrf_state or {"groups": [], "available_conditions": None}
    groups: list[dict[str, Any]] = list(state.get("groups", []))

    if 0 <= clicked_idx < len(groups):
        groups = [g for i, g in enumerate(groups) if i != clicked_idx]

    new_state: dict[str, Any] = {**state, "groups": groups}
    available = state.get("available_conditions")
    cards = _render_groups_list(groups, available)
    return cards, new_state


@callback(
    Output("hrf-runtime-state", "data", allow_duplicate=True),
    Input({"type": "hrf-rt-label", "group_idx": ALL}, "value"),
    Input({"type": "hrf-rt-cond",  "group_idx": ALL}, "value"),
    Input({"type": "hrf-rt-time",  "group_idx": ALL, "field": ALL}, "value"),
    State("hrf-runtime-state", "data"),
    prevent_initial_call=True,
)
def _hrf_sync_groups(
    labels: list[str | None],
    conditions: list[list[str] | None],
    times: list[float | None],
    hrf_state: dict[str, Any] | None,
) -> Any:
    """Sync Stage 1 DOM input values back into ``hrf-runtime-state``.

    Triggered by any label / conditions / time input change in Stage 1.
    Rebuilds the ``groups`` list from current DOM values.

    Notes
    -----
    Dash returns pattern-matching ``ALL`` inputs as a flat list ordered by DOM
    position.  With ``{"group_idx": ALL, "field": ALL}`` the flat order is
    ``[g0_tmin, g0_tmax, g0_bl_tmin, g0_bl_tmax, g1_tmin, ...]``
    (outer = group_idx, inner = field within the card render order).
    """
    if not labels and not conditions:
        return no_update

    state = hrf_state or {"groups": [], "available_conditions": None}
    n_groups = len(labels)
    n_time_fields = len(_TIME_FIELDS)

    def _f(v: Any, default: float) -> float:
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    groups: list[dict[str, Any]] = []
    for i in range(n_groups):
        t_offset = i * n_time_fields
        t_slice = times[t_offset : t_offset + n_time_fields] if times else []
        groups.append(
            {
                "label": labels[i] or "",
                "condition_names": list(
                    conditions[i] or [] if i < len(conditions) else []
                ),
                "tmin": _f(t_slice[0] if len(t_slice) > 0 else None, -2.0),
                "tmax": _f(t_slice[1] if len(t_slice) > 1 else None, 18.0),
                "baseline_tmin": _f(t_slice[2] if len(t_slice) > 2 else None, -2.0),
                "baseline_tmax": _f(t_slice[3] if len(t_slice) > 3 else None, 0.0),
            }
        )

    return {**state, "groups": groups}


@callback(
    Output("hrf-rt-stage1-container", "style"),
    Output("hrf-rt-stage2-container", "style"),
    Output("hrf-rt-footer-stage1", "style"),
    Output("hrf-rt-footer-stage2", "style"),
    Output(_STAGE_STORE_ID, "data"),
    Output("hrf-rt-stage2-container", "children"),
    Input("hrf-rt-next-btn", "n_clicks"),
    State("hrf-runtime-state", "data"),
    prevent_initial_call=True,
)
def _hrf_advance_to_stage2(
    n_clicks: int | None,
    hrf_state: dict[str, Any] | None,
) -> tuple[Any, ...]:
    """Advance from Stage 1 to Stage 2: render timings table from current groups."""
    no_op: tuple[Any, ...] = (
        no_update, no_update, no_update, no_update, no_update, no_update,
    )
    if not n_clicks:
        return no_op

    groups: list[dict[str, Any]] = (hrf_state or {}).get("groups", [])
    stage2_children = _render_stage2_body(groups)

    show = {"display": "block"}
    hide = {"display": "none"}
    return hide, show, hide, show, 2, stage2_children


@callback(
    Output("hrf-rt-stage1-container", "style", allow_duplicate=True),
    Output("hrf-rt-stage2-container", "style", allow_duplicate=True),
    Output("hrf-rt-footer-stage1", "style", allow_duplicate=True),
    Output("hrf-rt-footer-stage2", "style", allow_duplicate=True),
    Output(_STAGE_STORE_ID, "data", allow_duplicate=True),
    Input("hrf-rt-back-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _hrf_back_to_stage1(
    n_clicks: int | None,
) -> tuple[Any, ...]:
    """Return to Stage 1 from Stage 2."""
    no_op: tuple[Any, ...] = (
        no_update, no_update, no_update, no_update, no_update,
    )
    if not n_clicks:
        return no_op

    show = {"display": "block"}
    hide = {"display": "none"}
    return show, hide, show, hide, 1


@callback(
    Output("hrf-runtime-state", "data", allow_duplicate=True),
    Input({"type": "hrf-rt-time2", "group_idx": ALL, "field": ALL}, "value"),
    State("hrf-runtime-state", "data"),
    prevent_initial_call=True,
)
def _hrf_sync_stage2_times(
    times: list[float | None],
    hrf_state: dict[str, Any] | None,
) -> Any:
    """Sync Stage 2 time-input edits back into ``hrf-runtime-state``.

    Uses ``hrf-rt-time2`` IDs (distinct from Stage 1's ``hrf-rt-time``) to
    avoid duplicate-component conflicts when both stages are in the DOM.
    """
    if not times:
        return no_update

    state = hrf_state or {"groups": [], "available_conditions": None}
    groups: list[dict[str, Any]] = [dict(g) for g in state.get("groups", [])]
    n_time_fields = len(_TIME_FIELDS)
    n_groups = len(times) // n_time_fields if n_time_fields else 0
    field_names = [f for f, _ in _TIME_FIELDS]

    def _f(v: Any, default: float) -> float:
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    for i in range(min(n_groups, len(groups))):
        t_offset = i * n_time_fields
        t_slice = times[t_offset : t_offset + n_time_fields]
        for j, fname in enumerate(field_names):
            groups[i][fname] = _f(t_slice[j] if j < len(t_slice) else None, 0.0)

    return {**state, "groups": groups}


# ---------------------------------------------------------------------------
# Public helper — build params_override from hrf-runtime-state
# ---------------------------------------------------------------------------


def build_hrf_params_override(
    hrf_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert ``hrf-runtime-state`` store data into a ``params_override`` dict.

    Parameters
    ----------
    hrf_state:
        Current value of the ``"hrf-runtime-state"`` Dash store.

    Returns
    -------
    dict or None
        ``{"per_condition_groups": {...}, "_hrf_mode": "groups"}`` when at
        least one valid group exists; ``None`` otherwise.
        The ``_hrf_mode`` marker is stripped by ``advance_run`` before calling
        ``execute_current`` (``BlockAverageParams`` does not recognise it).
        Values are plain ``dict``s, coercible by
        ``BlockAverageParams.__post_init__`` to ``ConditionGroup`` instances.
    """
    if not hrf_state:
        return None
    groups: list[dict[str, Any]] = hrf_state.get("groups", [])
    if not groups:
        return None

    per_condition_groups: dict[str, Any] = {}
    for g in groups:
        label = (g.get("label") or "").strip()
        if not label:
            continue
        cond_names = [c for c in g.get("condition_names", []) if c]
        if not cond_names:
            continue
        per_condition_groups[label] = {
            "label": label,
            "condition_names": cond_names,
            "tmin": float(g.get("tmin", -2.0)),
            "tmax": float(g.get("tmax", 18.0)),
            "baseline_tmin": float(g.get("baseline_tmin", -2.0)),
            "baseline_tmax": float(g.get("baseline_tmax", 0.0)),
        }

    if not per_condition_groups:
        return None

    return {
        "per_condition_groups": per_condition_groups,
        "_hrf_mode": "groups",
    }
