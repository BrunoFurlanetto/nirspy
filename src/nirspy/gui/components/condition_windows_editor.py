"""Per-condition temporal window editor for BlockAverage.

Conditions are **always** sourced from the upstream LoadSnirf file —
the user is never asked to type condition names. When no usable SNIRF
is reachable the editor disables itself and tells the user to set the
LoadSnirf path first.

Layout: Switch ("Use per-condition windows") + read-only row per
condition with four numeric inputs (tmin, tmax, baseline_tmin,
baseline_tmax). No add / remove buttons — the row set is dictated by
the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
from dash import html

from nirspy.gui.components.param_metadata import metadata_for


def read_snirf_condition_names(snirf_path: str | None) -> list[str] | None:
    """Read stim condition names directly from a SNIRF file.

    Reads ``nirs/stim*/name`` via h5py without loading channel data, so it
    is cheap to call on every render. Returns ``None`` when the path is
    empty, missing, or unreadable.
    """
    if not snirf_path:
        return None
    p = Path(snirf_path)
    if not p.exists() or not p.is_file():
        return None
    try:
        import h5py
        import numpy as np
    except ImportError:
        return None
    names: list[str] = []
    try:
        with h5py.File(p, "r") as f:
            if "nirs" not in f:
                return None
            nirs = f["nirs"]
            stim_keys = sorted(
                k for k in nirs if k.startswith("stim")
            )
            for key in stim_keys:
                grp = nirs[key]
                if "name" not in grp:
                    continue
                raw = np.array(grp["name"]).ravel()
                if raw.size == 0:
                    continue
                val = raw.item() if raw.size == 1 else raw[0]
                name = (
                    val.decode("utf-8", errors="replace")
                    if isinstance(val, bytes)
                    else str(val)
                ).strip()
                if name:
                    names.append(name)
    except (OSError, KeyError, ValueError):
        return None
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique or None


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
) -> dbc.Row:
    """Render one read-only-name row with four numeric window inputs.

    Uses ``debounce=True`` so the value callback fires only on blur /
    Enter -- this prevents the re-render cycle from resetting the input
    while the user is still typing.

    Defaults to the **maximum** of the allowed range (via ParamMeta) so
    the user starts from the top and adjusts downward.
    """
    name_cell = html.Div(
        html.Span(condition, className="fw-semibold small"),
        className="d-flex align-items-center h-100",
    )

    fields = ["tmin", "tmax", "baseline_tmin", "baseline_tmax"]
    inputs: list[dbc.Col] = []
    for f in fields:
        meta = metadata_for("block_average", f)
        attrs: dict[str, Any] = {}
        default: float = 0.0
        if meta is not None:
            if meta.min is not None:
                attrs["min"] = meta.min
            if meta.max is not None:
                attrs["max"] = meta.max
                default = float(meta.max)
            if meta.step is not None:
                attrs["step"] = meta.step
        inputs.append(
            dbc.Col(
                dbc.Input(
                    id=_row_id(instance_id, condition, f),
                    type="number",
                    value=values.get(f, default),
                    size="sm",
                    debounce=True,
                    **attrs,
                ),
                width=2,
            )
        )

    return dbc.Row(
        [dbc.Col(name_cell, width=4)] + inputs,
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
        Condition names harvested from the upstream LoadSnirf file. When
        ``None`` (or empty) the editor renders disabled with a hint.
    """
    current_value = current_value or {}
    has_snirf = bool(available_conditions)
    is_enabled = has_snirf and len(current_value) > 0

    switch = dbc.Switch(
        id={"type": "cond-window-switch", "instance_id": instance_id},
        label="Use per-condition windows",
        value=is_enabled,
        disabled=not has_snirf,
        className="mb-2",
    )

    if not has_snirf:
        return html.Div(
            [
                html.Label(
                    "Per-condition windows",
                    className="small fw-bold mb-1",
                ),
                switch,
                html.Small(
                    "Set the LoadSnirf path first — conditions are read "
                    "from the .snirf file and cannot be typed manually.",
                    className="text-muted d-block mb-1",
                ),
            ],
            className="mb-3 p-2 border rounded",
        )

    header = dbc.Row(
        [
            dbc.Col(html.Small("Condition", className="fw-bold"), width=4),
            dbc.Col(html.Small("tmin", className="fw-bold"), width=2),
            dbc.Col(html.Small("tmax", className="fw-bold"), width=2),
            dbc.Col(html.Small("bl_tmin", className="fw-bold"), width=2),
            dbc.Col(html.Small("bl_tmax", className="fw-bold"), width=2),
        ],
        className="mb-1 g-1",
    )

    rows = []
    for cond in available_conditions or []:
        win = current_value.get(cond)
        vals: dict[str, float] = {}
        if win is None:
            pass
        elif isinstance(win, dict):
            vals = win
        elif hasattr(win, "tmin"):
            vals = {
                "tmin": win.tmin,
                "tmax": win.tmax,
                "baseline_tmin": win.baseline_tmin,
                "baseline_tmax": win.baseline_tmax,
            }
        rows.append(_render_row(instance_id, cond, vals))

    hint = html.Small(
        f"{len(available_conditions or [])} condition(s) loaded from SNIRF — "
        "edit the temporal windows below.",
        className="text-muted d-block mb-1",
    )

    table_div = html.Div(
        [header] + rows,
        id={"type": "cond-window-table", "instance_id": instance_id},
        style={"display": "block" if is_enabled else "none"},
    )

    return html.Div(
        [
            html.Label(
                "Per-condition windows",
                className="small fw-bold mb-1",
            ),
            switch,
            hint,
            table_div,
        ],
        className="mb-3 p-2 border rounded",
    )
