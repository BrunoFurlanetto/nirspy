"""Builder-side editor for per-condition groups (T-025).

Renders a static editor (no runtime dialog) that lets the user
configure ``BlockAverageParams.per_condition_groups`` in the builder.

Layout
------
- A ``[+ Add group]`` button at the top creates a new empty group card.
- Each group card shows:
    - A text input for the group **label**.
    - A multi-select ``dcc.Dropdown`` for **condition_names** (drawn from
      the upstream LoadSnirf SNIRF file — same source as T-012).
    - Four numeric inputs: ``tmin``, ``tmax``, ``baseline_tmin``,
      ``baseline_tmax``.
    - A ``[- Remove]`` button.
- Below all group cards, an optional warning lists conditions that are
  not assigned to any group (orphan conditions).

Component IDs
-------------
All pattern-matching IDs follow the ``{"type": ..., "instance_id": ...,
"group_idx": ..., "field": ...}`` convention so that callbacks can
identify which group and which field triggered a change.

Invariants
----------
- A condition may appear in **at most one group**; duplicates cause the
  warning box to highlight the conflicting conditions.
- The editor does *not* mutate pipeline-state directly; it merely
  renders the current state.  Mutation happens in the callbacks module.
- When ``available_conditions`` is ``None``, the multiselect dropdowns
  still render (users may type manually) but an informational hint is
  shown.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from nirspy.gui.components.condition_timeline import render_condition_timeline
from nirspy.gui.components.param_metadata import metadata_for

# ---------------------------------------------------------------------------
# Public ID builders (used by callbacks to identify components)
# ---------------------------------------------------------------------------


def group_label_id(instance_id: str, group_idx: int) -> dict[str, Any]:
    """Pattern-matching ID for the group label text input."""
    return {
        "type": "cg-label",
        "instance_id": instance_id,
        "group_idx": group_idx,
    }


def group_conditions_id(instance_id: str, group_idx: int) -> dict[str, Any]:
    """Pattern-matching ID for the conditions multiselect Dropdown."""
    return {
        "type": "cg-conditions",
        "instance_id": instance_id,
        "group_idx": group_idx,
    }


def group_time_id(
    instance_id: str, group_idx: int, field: str
) -> dict[str, Any]:
    """Pattern-matching ID for a tmin/tmax/baseline numeric input."""
    return {
        "type": "cg-time",
        "instance_id": instance_id,
        "group_idx": group_idx,
        "field": field,
    }


def group_remove_id(instance_id: str, group_idx: int) -> dict[str, Any]:
    """Pattern-matching ID for the [- Remove] button of a group card."""
    return {
        "type": "cg-remove",
        "instance_id": instance_id,
        "group_idx": group_idx,
    }


def add_group_btn_id(instance_id: str) -> dict[str, Any]:
    """Pattern-matching ID for the [+ Add group] button."""
    return {"type": "cg-add", "instance_id": instance_id}


def groups_mode_radio_id(instance_id: str) -> dict[str, Any]:
    """Pattern-matching ID for the mode radio toggle."""
    return {"type": "cg-mode-radio", "instance_id": instance_id}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _time_defaults(params_snapshot: dict[str, Any]) -> dict[str, float]:
    """Return global tmin/tmax/baseline_tmin/baseline_tmax from params."""
    def _f(key: str, default: float) -> float:
        v = params_snapshot.get(key)
        return float(v) if v is not None else default

    return {
        "tmin": _f("tmin", -2.0),
        "tmax": _f("tmax", 18.0),
        "baseline_tmin": _f("baseline_tmin", -2.0),
        "baseline_tmax": _f("baseline_tmax", 0.0),
    }


def _numeric_attrs_for(field_name: str, block_id: str = "block_average") -> dict[str, Any]:
    """Return HTML5 min/max/step attrs from ParamMeta for a given block.

    Parameters
    ----------
    field_name:
        The dataclass field name (e.g. ``"tmin"``).
    block_id:
        Registry ID of the owning block.  Defaults to ``"block_average"``
        for backward compatibility.
    """
    meta = metadata_for(block_id, field_name)
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


def _render_group_card(
    instance_id: str,
    group_idx: int,
    label: str,
    condition_names: list[str],
    event_indices: list[int],
    time_vals: dict[str, float],
    available_conditions: list[str] | None,
    duplicated_conditions: set[str],
    block_id: str = "block_average",
) -> dbc.Card:
    """Render a single condition-group card."""
    options = (
        [{"label": c, "value": c} for c in available_conditions]
        if available_conditions
        else []
    )
    # Highlight duplicated selections in red
    invalid = [c for c in condition_names if c in duplicated_conditions]
    dropdown_style: dict[str, Any] = (
        {"border": "1px solid var(--bs-danger)"} if invalid else {}
    )

    time_fields = [
        ("tmin", "tmin"),
        ("tmax", "tmax"),
        ("baseline_tmin", "bl_tmin"),
        ("baseline_tmax", "bl_tmax"),
    ]
    time_inputs = []
    for field_name, label_short in time_fields:
        attrs = _numeric_attrs_for(field_name, block_id=block_id)
        if "step" not in attrs:
            attrs["step"] = 0.1
        time_inputs.append(
            dbc.Col(
                [
                    html.Small(label_short, className="fw-bold d-block"),
                    dbc.Input(
                        id=group_time_id(instance_id, group_idx, field_name),
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

    # Timeline-first: card always shows badge + hidden dropdown for callback compat.
    # Users select occurrences via the shared timeline above.
    if event_indices:
        conditions_section: list[Any] = [
            html.Small("Occurrences (timeline)", className="fw-bold d-block mb-1"),
            dbc.Badge(
                f"{len(event_indices)} occurrence(s) selected",
                color="primary",
                className="mb-1",
            ),
            html.Small(
                f"Indices: {event_indices}",
                className="text-muted d-block",
            ),
            dcc.Dropdown(
                id=group_conditions_id(instance_id, group_idx),
                options=[],
                value=[],
                multi=True,
                style={"display": "none"},
            ),
        ]
    else:
        conditions_section = [
            html.Small("Occurrences", className="fw-bold d-block mb-1"),
            html.Small(
                "Set this group as Active above, then click markers on the timeline.",
                className="text-muted d-block mb-1",
            ),
            dcc.Dropdown(
                id=group_conditions_id(instance_id, group_idx),
                options=options,
                value=condition_names,
                multi=True,
                placeholder="(or pick by name — legacy mode)",
                style={**dropdown_style, "display": "none" if available_conditions else "block"},
            ),
            html.Div(
                [
                    html.Small(
                        f"Conflict: {', '.join(sorted(invalid))} "
                        "already in another group.",
                        className="text-danger d-block mt-1",
                    )
                ]
                if invalid
                else [],
            ),
        ]

    card_body = dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Small("Group label", className="fw-bold d-block"),
                            dbc.Input(
                                id=group_label_id(instance_id, group_idx),
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
                            id=group_remove_id(instance_id, group_idx),
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
            *conditions_section,
            dbc.Row(
                time_inputs,
                className="mt-2 g-1",
            ),
        ],
        className="p-2",
    )

    return dbc.Card(card_body, className="mb-2")


def _collect_orphans(
    available_conditions: list[str] | None,
    groups: list[dict[str, Any]],
) -> list[str]:
    """Return conditions in *available_conditions* not in any group."""
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
# Public render function
# ---------------------------------------------------------------------------


def render_condition_groups_editor(
    instance_id: str,
    current_groups: dict[str, Any] | None,
    available_conditions: list[str] | None = None,
    snirf_path: str | None = None,
    active_group_label: str | None = None,
    block_id: str = "block_average",
) -> html.Div:
    """Render the static condition-groups editor for the builder.

    Parameters
    ----------
    instance_id:
        Unique pipeline step ID (matches the BlockAverage step's
        ``instance_id`` in ``pipeline-state``).
    current_groups:
        Current ``per_condition_groups`` dict.  Keys are group labels;
        values are ``ConditionGroup``-like objects or plain dicts with
        fields ``condition_names``, ``event_indices``, ``tmin``, ``tmax``,
        ``baseline_tmin``, ``baseline_tmax``.
    available_conditions:
        Condition names sourced from the upstream LoadSnirf file.
        ``None`` when no SNIRF is reachable yet.
    snirf_path:
        Absolute path to the SNIRF file (for the timeline chart).
        ``None`` hides the timeline and shows a placeholder.
    active_group_label:
        Currently active group for click-to-toggle on the timeline.
        ``None`` disables toggle.
    block_id:
        Registry ID of the owning block.  Determines which ParamMeta
        entries are used for min/max/step on numeric inputs inside each
        group card.  Defaults to ``"block_average"`` for backward
        compatibility.

    Returns
    -------
    html.Div
        The full groups editor widget including the condition timeline.
    """
    cg = current_groups or {}

    # Normalise to list of plain dicts for uniform rendering
    groups_list: list[dict[str, Any]] = []
    for grp_label, grp_val in cg.items():
        if hasattr(grp_val, "condition_names"):
            # ConditionGroup dataclass instance
            groups_list.append(
                {
                    "label": grp_label,
                    "condition_names": list(grp_val.condition_names),
                    "event_indices": list(grp_val.event_indices),
                    "tmin": grp_val.tmin,
                    "tmax": grp_val.tmax,
                    "baseline_tmin": grp_val.baseline_tmin,
                    "baseline_tmax": grp_val.baseline_tmax,
                }
            )
        elif isinstance(grp_val, dict):
            _t = grp_val.get("tmin")
            _x = grp_val.get("tmax")
            _bm = grp_val.get("baseline_tmin")
            _bx = grp_val.get("baseline_tmax")
            groups_list.append(
                {
                    "label": grp_label,
                    "condition_names": list(grp_val.get("condition_names") or []),
                    "event_indices": list(grp_val.get("event_indices") or []),
                    "tmin": float(_t) if _t is not None else -2.0,
                    "tmax": float(_x) if _x is not None else 18.0,
                    "baseline_tmin": float(_bm) if _bm is not None else -2.0,
                    "baseline_tmax": float(_bx) if _bx is not None else 0.0,
                }
            )

    duplicated = _collect_duplicates(groups_list)

    # Timeline widget (above the group cards)
    timeline_widget = render_condition_timeline(
        instance_id=instance_id,
        snirf_path=snirf_path,
        groups_state=cg,
        active_group_label=active_group_label,
    )

    # Build group cards
    cards: list[Any] = []
    for idx, g in enumerate(groups_list):
        cards.append(
            _render_group_card(
                instance_id=instance_id,
                group_idx=idx,
                label=g["label"],
                condition_names=g["condition_names"],
                event_indices=g["event_indices"],
                time_vals={
                    "tmin": g["tmin"],
                    "tmax": g["tmax"],
                    "baseline_tmin": g["baseline_tmin"],
                    "baseline_tmax": g["baseline_tmax"],
                },
                available_conditions=available_conditions,
                duplicated_conditions=duplicated,
                block_id=block_id,
            )
        )

    add_btn = dbc.Button(
        "+ Add group",
        id=add_group_btn_id(instance_id),
        color="primary",
        outline=True,
        size="sm",
        className="mb-2",
    )

    # Orphan conditions warning
    orphans = _collect_orphans(available_conditions, groups_list)
    orphan_warn: list[Any] = []
    if orphans:
        orphan_warn = [
            html.Div(
                html.Small(
                    f"Unassigned conditions: {', '.join(sorted(orphans))}",
                    className="text-secondary",
                ),
                className="mt-1 p-1 border rounded border-secondary",
                id={"type": "cg-orphan-warn", "instance_id": instance_id},
            )
        ]

    no_groups_hint: list[Any] = []
    if not groups_list:
        no_groups_hint = [
            html.Small(
                "No groups defined. Click [+ Add group] to create one.",
                className="text-muted d-block mb-2",
            )
        ]

    return html.Div(
        [
            timeline_widget,
            *no_groups_hint,
            *cards,
            add_btn,
            *orphan_warn,
        ],
        id={"type": "cg-editor-container", "instance_id": instance_id},
    )
