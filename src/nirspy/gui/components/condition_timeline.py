"""Condition timeline selector for BlockAverage groups (T-030).

Renders an interactive Plotly scatter chart where each stimulus occurrence
is a clickable marker.  Users click markers to add or remove individual
occurrences from the *active* condition group, enabling sub-selection of
specific trials within a condition.

Layout
------
- X axis: onset time in seconds.
- Y axis: condition name (categorical, one row per stim type).
- Markers coloured by membership:
  - Dark grey: not assigned to any group.
  - Accent colour of the active group: member of the active group.
  - Distinct lighter colour per other group: member of another group.
- ``customdata[0]``: chronological event index (int), used by the click
  callback in ``param_callbacks.py`` to toggle membership.

Public API
----------
``render_condition_timeline(snirf_path, groups_state, active_group_label)``
    Returns a ``dcc.Graph`` (or a placeholder ``html.Div`` when no SNIRF
    is available).

``condition_timeline_id(instance_id)``
    Returns the pattern-matching component ID dict.

``active_group_selector_id(instance_id)``
    Returns the pattern-matching component ID dict for the active-group
    radio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import dcc, html

# ---------------------------------------------------------------------------
# Colour palette for groups
# ---------------------------------------------------------------------------

# Index 0  -> active group accent (prominent)
# Index 1+ -> other groups (lighter, distinct)
_GROUP_COLORS = [
    "#0d6efd",  # Bootstrap primary blue  -- active group (index 0 = active)
    "#198754",  # Bootstrap success green
    "#dc3545",  # Bootstrap danger red
    "#fd7e14",  # Bootstrap orange
    "#6f42c1",  # Bootstrap purple
    "#20c997",  # Bootstrap teal
    "#ffc107",  # Bootstrap yellow
]
_UNASSIGNED_COLOR = "#adb5bd"  # Bootstrap secondary grey


# ---------------------------------------------------------------------------
# Public ID builders
# ---------------------------------------------------------------------------


def condition_timeline_id(instance_id: str) -> dict[str, Any]:
    """Pattern-matching ID for the condition-timeline dcc.Graph."""
    return {"type": "condition-timeline-graph", "instance_id": instance_id}


def active_group_selector_id(instance_id: str) -> dict[str, Any]:
    """Pattern-matching ID for the active-group radio/dropdown selector."""
    return {"type": "condition-timeline-active-group", "instance_id": instance_id}


# ---------------------------------------------------------------------------
# SNIRF event reader (all occurrences, not just unique names)
# ---------------------------------------------------------------------------


def _read_snirf_events(
    path: str | Path,
) -> list[tuple[int, str, float]]:
    """Read all stimulus events from a SNIRF file via h5py.

    Returns a list of ``(event_index, condition_name, onset_seconds)``
    tuples, sorted by onset (i.e. the chronological order that MNE will
    assign event_indices to).  Returns an empty list when the file is
    unavailable or has no stim groups.

    The ``event_index`` is the 0-based position in the sorted list —
    matching the convention used by
    :meth:`~nirspy.engine.mne_adapter.MNEAdapter._events_by_indices`.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists() or not p.is_file():
        return []
    try:
        import h5py
        import numpy as np
    except ImportError:
        return []

    raw_events: list[tuple[str, float]] = []  # (name, onset)
    try:
        with h5py.File(p, "r") as f:
            if "nirs" not in f:
                return []
            nirs = f["nirs"]
            stim_keys = sorted(k for k in nirs if k.startswith("stim"))
            for key in stim_keys:
                grp = nirs[key]
                if "name" not in grp or "data" not in grp:
                    continue
                raw_name_arr = np.array(grp["name"]).ravel()
                if raw_name_arr.size == 0:
                    continue
                name_val = (
                    raw_name_arr.item() if raw_name_arr.size == 1
                    else raw_name_arr[0]
                )
                cond_name: str = (
                    name_val.decode("utf-8", errors="replace")
                    if isinstance(name_val, bytes)
                    else str(name_val)
                ).strip()
                if not cond_name:
                    continue

                data_arr = np.array(grp["data"])
                if data_arr.ndim == 1:
                    # Single occurrence: data = [onset, duration, value]
                    onset = float(data_arr[0])
                    raw_events.append((cond_name, onset))
                else:
                    # Multiple occurrences: rows = [onset, duration, value]
                    for row in data_arr:
                        onset = float(row[0])
                        raw_events.append((cond_name, onset))
    except (OSError, KeyError, ValueError):
        return []

    # Sort by onset to establish chronological event_index order
    raw_events.sort(key=lambda e: e[1])
    return [
        (idx, name, onset)
        for idx, (name, onset) in enumerate(raw_events)
    ]


# ---------------------------------------------------------------------------
# Colour assignment helpers
# ---------------------------------------------------------------------------


def _build_color_map(
    groups_state: dict[str, Any] | None,
    active_group_label: str | None,
) -> dict[str, str]:
    """Return a mapping of group_label -> hex colour.

    The active group always gets index-0 colour.  Other groups get
    subsequent palette entries.
    """
    if not groups_state:
        return {}
    labels = list(groups_state.keys())
    # Put active group first so it gets color index 0
    ordered: list[str] = []
    if active_group_label and active_group_label in labels:
        ordered.append(active_group_label)
    for lbl in labels:
        if lbl not in ordered:
            ordered.append(lbl)
    return {
        lbl: _GROUP_COLORS[i % len(_GROUP_COLORS)]
        for i, lbl in enumerate(ordered)
    }


def _event_index_to_group(
    groups_state: dict[str, Any] | None,
) -> dict[int, str]:
    """Return mapping of event_index -> group_label for quick lookup."""
    result: dict[int, str] = {}
    if not groups_state:
        return result
    for lbl, grp_val in groups_state.items():
        indices: list[int] = []
        if isinstance(grp_val, dict):
            indices = list(grp_val.get("event_indices") or [])
        elif hasattr(grp_val, "event_indices"):
            indices = list(grp_val.event_indices)
        for idx in indices:
            result[idx] = lbl
    return result


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------


def render_condition_timeline(
    instance_id: str,
    snirf_path: str | None,
    groups_state: dict[str, Any] | None,
    active_group_label: str | None,
) -> html.Div:
    """Render the condition-timeline graph + active-group selector.

    Parameters
    ----------
    instance_id:
        Unique pipeline step ID (matches the BlockAverage step's
        ``instance_id`` in ``pipeline-state``).
    snirf_path:
        Absolute path to the SNIRF file being processed, or ``None``.
    groups_state:
        Current ``per_condition_groups`` dict from ``pipeline-state``.
    active_group_label:
        Which group label is currently selected for click-to-toggle.
        ``None`` means "no group selected" and clicks are no-ops.

    Returns
    -------
    html.Div
        Container holding (a) the active-group radio selector and (b)
        the ``dcc.Graph`` with the timeline scatter plot.
    """
    events = _read_snirf_events(snirf_path) if snirf_path else []

    if not events:
        return html.Div(
            [
                html.Small(
                    "Set the LoadSnirf path to load the condition timeline.",
                    className="text-muted d-block",
                )
            ],
            id={"type": "condition-timeline-wrapper", "instance_id": instance_id},
        )

    color_map = _build_color_map(groups_state, active_group_label)
    idx_to_group = _event_index_to_group(groups_state)

    # Build per-event data for scatter
    x_vals: list[float] = []
    y_vals: list[str] = []
    colors: list[str] = []
    customdata: list[list[Any]] = []
    symbols: list[str] = []
    hover_texts: list[str] = []

    for ev_idx, cond_name, onset in events:
        grp_label = idx_to_group.get(ev_idx)
        if grp_label is None:
            color = _UNASSIGNED_COLOR
            symbol = "circle"
            hover = f"#{ev_idx} · {cond_name} · {onset:.2f}s — unassigned"
        elif grp_label == active_group_label:
            color = color_map.get(grp_label, _GROUP_COLORS[0])
            symbol = "circle"
            hover = f"#{ev_idx} · {cond_name} · {onset:.2f}s — {grp_label} (active)"
        else:
            color = color_map.get(grp_label, _GROUP_COLORS[1])
            symbol = "diamond"
            hover = f"#{ev_idx} · {cond_name} · {onset:.2f}s — {grp_label}"

        x_vals.append(onset)
        y_vals.append(cond_name)
        colors.append(color)
        customdata.append([ev_idx])
        symbols.append(symbol)
        hover_texts.append(hover)

    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker={
                "color": colors,
                "size": 12,
                "symbol": symbols,
                "line": {"width": 1, "color": "white"},
            },
            customdata=customdata,
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=max(120, 50 * len({e[1] for e in events})),
        xaxis_title="Onset (s)",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        yaxis={"type": "category", "autorange": "reversed"},
    )

    # Active-group radio options
    group_labels = list((groups_state or {}).keys())
    radio_options = [{"label": lbl, "value": lbl} for lbl in group_labels]
    radio_options.insert(0, {"label": "— none —", "value": "__none__"})
    radio_value = active_group_label if active_group_label else "__none__"

    import dash_bootstrap_components as dbc

    selector = dbc.RadioItems(
        id=active_group_selector_id(instance_id),
        options=radio_options,
        value=radio_value,
        inline=True,
        className="mb-1",
    )

    graph = dcc.Graph(
        id=condition_timeline_id(instance_id),
        figure=fig,
        config={"displayModeBar": False},
        style={"cursor": "pointer"},
    )

    return html.Div(
        [
            html.Small("Active group for click-to-select:", className="fw-bold d-block mb-1"),
            selector,
            graph,
        ],
        id={"type": "condition-timeline-wrapper", "instance_id": instance_id},
        className="mb-2",
    )
