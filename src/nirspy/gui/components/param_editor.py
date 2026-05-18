"""Parameter editor component -- auto-generated form from dataclass fields.

Uses :func:`dataclasses.fields` to introspect the ``params_class`` declared
on a block's :class:`~nirspy.domain.block.BlockSpec` and renders a
Bootstrap form with appropriate input widgets per field type.

No per-block form is hardcoded; everything is derived from the dataclass
definition (ADR-007).

When a :class:`~nirspy.gui.components.param_metadata.ParamMeta` entry exists
for a ``(block_id, field_name)`` pair the editor renders a rich label
(with unit), a tooltip (description + reference) and HTML5 min/max/step
attributes.  ``Optional[float]`` fields get a "use default" checkbox.
``list[str]`` fields render a multi-select dropdown when channel names are
available, with a text-input fallback otherwise.

Note: because many block modules use ``from __future__ import annotations``,
``field.type`` is often a *string* rather than an actual type object.  The
helper :func:`_resolve_field_type` normalises both representations.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Union, get_args, get_origin

import dash_bootstrap_components as dbc
from dash import dcc, html

from nirspy.gui.components.condition_windows_editor import (
    render_condition_windows_editor,
)
from nirspy.gui.components.param_metadata import ParamMeta, metadata_for

# Mapping from stringified annotations to concrete types
_STR_TYPE_MAP: dict[str, type[Any]] = {
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
}


def _resolve_field_type(tp: Any) -> tuple[bool, type[Any]]:
    """Return ``(is_optional, resolved_type)`` for a dataclass field type.

    Handles both real type objects and stringified annotations
    (caused by ``from __future__ import annotations``).
    """
    # --- string annotations ---
    if isinstance(tp, str):
        clean = tp.replace(" ", "")
        is_opt = False
        # "float|None", "None|float", "list[str]|None", etc.
        if "|" in clean:
            parts = [p for p in clean.split("|") if p.lower() != "none"]
            is_opt = len(parts) < len(clean.split("|"))
            if len(parts) == 1:
                inner = parts[0]
                if inner.startswith("list[") and inner.endswith("]"):
                    return is_opt, list
                return is_opt, _STR_TYPE_MAP.get(inner, str)
            return is_opt, str
        if clean.startswith("list[") and clean.endswith("]"):
            return False, list
        return False, _STR_TYPE_MAP.get(clean, str)

    # --- real type objects ---
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            inner = non_none[0]
            inner_origin = get_origin(inner)
            if inner_origin is list:
                return True, list
            return True, inner

    if origin is list:
        return False, list

    if tp is bool:
        return False, bool
    if tp is int:
        return False, int
    if tp is float:
        return False, float
    if tp is str:
        return False, str

    return False, str



def _make_label(field_name: str, meta: ParamMeta | None) -> str:
    """Build the display label from metadata or raw field name."""
    if meta is None:
        return field_name
    label = meta.label
    if meta.unit:
        label = f"{label} ({meta.unit})"
    return label


def _make_tooltip(
    meta: ParamMeta | None,
    target_id: dict[str, str],
) -> dbc.Tooltip | None:
    """Build a ``dbc.Tooltip`` if metadata has a description."""
    if meta is None or not meta.description:
        return None
    text = meta.description
    if meta.reference:
        text = f"{text}\n\nRef: {meta.reference}"
    return dbc.Tooltip(
        text,
        target=target_id,
        placement="right",
        delay={"show": 300, "hide": 100},
    )


def _numeric_attrs(meta: ParamMeta | None) -> dict[str, Any]:
    """Extract min/max/step kwargs for ``dbc.Input`` from metadata."""
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

def _field_to_input(
    field: dataclasses.Field[Any],
    value: Any,
    instance_id: str,
    block_id: str,
    *,
    available_channels: list[str] | None = None,
    available_conditions: list[str] | None = None,
) -> list[Any]:
    """Convert a single dataclass field to form row(s).

    Returns a **list** of components (usually one ``dbc.Row`` plus an
    optional ``dbc.Tooltip``).
    """
    field_name = field.name

    # Surgical delegation for per_condition_windows (T-012)
    if block_id == "block_average" and field_name == "per_condition_windows":
        editor = render_condition_windows_editor(
            instance_id,
            value if isinstance(value, dict) else None,
            available_conditions=available_conditions,
        )
        return [editor]

    input_id: dict[str, str] = {
        "type": "param-input",
        "instance_id": instance_id,
        "field": field_name,
    }

    meta = metadata_for(block_id, field_name)
    label_text = _make_label(field_name, meta)
    tooltip = _make_tooltip(meta, input_id)

    is_optional, resolved = _resolve_field_type(field.type)

    components: list[Any] = []

    # --- list[str] (multiselect / text fallback) ---------------------------
    if resolved is list:
        current_list: list[str] = value if isinstance(value, list) else []
        if available_channels:
            ctrl: Any = dcc.Dropdown(
                id=input_id,
                options=[{"label": ch, "value": ch} for ch in available_channels],
                value=current_list,
                multi=True,
                placeholder="Select channels...",
            )
        else:
            ctrl = dbc.Input(
                id=input_id,
                type="text",
                value=", ".join(current_list) if current_list else "",
                placeholder="Run pipeline first to load channel list",
                size="sm",
            )
        row = dbc.Row(
            [
                dbc.Label(label_text, width=5, className="small"),
                dbc.Col(ctrl, width=7),
            ],
            className="mb-2",
        )
        components.append(row)
        if tooltip:
            components.append(tooltip)
        return components

    # --- bool ---------------------------------------------------------------
    if resolved is bool:
        ctrl = dbc.Checkbox(
            id=input_id,
            value=bool(value) if value is not None else False,
            className="ms-2",
        )
        row = dbc.Row(
            [
                dbc.Label(label_text, width=5, className="small"),
                dbc.Col(ctrl, width=7),
            ],
            className="mb-2 align-items-center",
        )
        components.append(row)
        if tooltip:
            components.append(tooltip)
        return components

    # --- int ----------------------------------------------------------------
    if resolved is int:
        num_attrs = _numeric_attrs(meta)
        if "step" not in num_attrs:
            num_attrs["step"] = 1
        ctrl = dbc.Input(
            id=input_id,
            type="number",
            value=value if value is not None else "",
            size="sm",
            **num_attrs,
        )
        row = dbc.Row(
            [
                dbc.Label(label_text, width=5, className="small"),
                dbc.Col(ctrl, width=7),
            ],
            className="mb-2",
        )
        components.append(row)
        if tooltip:
            components.append(tooltip)
        return components

    # --- Optional[float] with "use default" checkbox -----------------------
    if is_optional and resolved is float:
        use_default = value is None
        checkbox_id: dict[str, str] = {
            "type": "param-optional-toggle",
            "instance_id": instance_id,
            "field": field_name,
        }
        check = dbc.Checkbox(
            id=checkbox_id,
            value=use_default,
            label="use default",
            className="small",
        )
        num_attrs = _numeric_attrs(meta)
        if "step" not in num_attrs:
            num_attrs["step"] = 0.01
        ctrl = dbc.Input(
            id=input_id,
            type="number",
            value=value if value is not None else "",
            disabled=use_default,
            size="sm",
            **num_attrs,
        )
        row = dbc.Row(
            [
                dbc.Label(label_text, width=5, className="small"),
                dbc.Col([check, ctrl], width=7),
            ],
            className="mb-2",
        )
        components.append(row)
        if tooltip:
            components.append(tooltip)
        return components

    # --- float (including non-optional) ------------------------------------
    if resolved is float:
        num_attrs = _numeric_attrs(meta)
        if "step" not in num_attrs:
            num_attrs["step"] = 0.01
        ctrl = dbc.Input(
            id=input_id,
            type="number",
            value=value if value is not None else "",
            size="sm",
            **num_attrs,
        )
        row = dbc.Row(
            [
                dbc.Label(label_text, width=5, className="small"),
                dbc.Col(ctrl, width=7),
            ],
            className="mb-2",
        )
        components.append(row)
        if tooltip:
            components.append(tooltip)
        return components

    # --- str or fallback ----------------------------------------------------
    ctrl = dbc.Input(
        id=input_id,
        type="text",
        value=str(value) if value is not None else "",
        size="sm",
    )
    row = dbc.Row(
        [
            dbc.Label(label_text, width=5, className="small"),
            dbc.Col(ctrl, width=7),
        ],
        className="mb-2",
    )
    components.append(row)
    if tooltip:
        components.append(tooltip)
    return components


def render_param_editor(
    block_id: str | None,
    instance_id: str | None,
    params_class: type[Any] | None,
    current_values: dict[str, Any],
    *,
    available_channels: list[str] | None = None,
    available_conditions: list[str] | None = None,
) -> html.Div:
    """Auto-generate a parameter form for *params_class*.

    Parameters
    ----------
    block_id:
        Registry ID of the selected block (for display purposes).
    instance_id:
        The unique session ID of the selected pipeline step.
    params_class:
        Dataclass *type* holding the block parameters, or *None*.
    current_values:
        Dict of current parameter values.
    available_channels:
        Optional list of channel names from the last Raw in the pipeline
        results.  Used by ``list[str]`` fields to populate a multi-select
        dropdown.

    Returns
    -------
    html.Div
        Form container, or a placeholder when no block is selected or
        the block has no parameters.
    """
    if block_id is None or instance_id is None:
        return html.Div(
            html.P(
                "Select a block to edit its parameters.",
                className="text-muted small",
            )
        )

    if params_class is None or not dataclasses.is_dataclass(params_class):
        return html.Div(
            [
                html.H6(block_id, className="mb-2"),
                html.P("No parameters", className="text-muted small"),
            ]
        )

    fields = dataclasses.fields(params_class)
    if not fields:
        return html.Div(
            [
                html.H6(block_id, className="mb-2"),
                html.P("No parameters", className="text-muted small"),
            ]
        )

    children: list[Any] = []
    for f in fields:
        default = f.default if f.default is not dataclasses.MISSING else None
        val = current_values.get(f.name, default)
        children.extend(
            _field_to_input(
                f,
                val,
                instance_id,
                block_id,
                available_channels=available_channels,
                available_conditions=available_conditions,
            )
        )

    return html.Div(
        [
            html.H6(block_id, className="mb-3"),
            dbc.Form(children),
        ]
    )
