"""Parameter editor component -- auto-generated form from dataclass fields.

Uses :func:`dataclasses.fields` to introspect the ``params_class`` declared
on a block's :class:`~nirspy.domain.block.BlockSpec` and renders a
Bootstrap form with appropriate input widgets per field type.

No per-block form is hardcoded; everything is derived from the dataclass
definition (ADR-007).

Note: because many block modules use ``from __future__ import annotations``,
``field.type`` is often a *string* rather than an actual type object.  The
helper :func:`_resolve_field_type` normalises both representations.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Union, get_args, get_origin

import dash_bootstrap_components as dbc
from dash import html

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
                return is_opt, _STR_TYPE_MAP.get(parts[0], str)
            return is_opt, str
        return False, _STR_TYPE_MAP.get(clean, str)

    # --- real type objects ---
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return True, non_none[0]

    if tp is bool:
        return False, bool
    if tp is int:
        return False, int
    if tp is float:
        return False, float
    if tp is str:
        return False, str

    return False, str


def _field_to_input(
    field: dataclasses.Field[Any],
    value: Any,
    instance_id: str,
) -> dbc.Row:
    """Convert a single dataclass field to a form row."""
    field_name = field.name
    input_id = {"type": "param-input", "instance_id": instance_id, "field": field_name}

    _optional, resolved = _resolve_field_type(field.type)

    # --- bool ---------------------------------------------------------------
    if resolved is bool:
        control: Any = dbc.Checkbox(
            id=input_id,
            value=bool(value) if value is not None else False,
            className="ms-2",
        )
        return dbc.Row(
            [
                dbc.Label(field_name, width=5, className="small"),
                dbc.Col(control, width=7),
            ],
            className="mb-2 align-items-center",
        )

    # --- int ----------------------------------------------------------------
    if resolved is int:
        ctrl = dbc.Input(
            id=input_id,
            type="number",
            step=1,
            value=value if value is not None else "",
            size="sm",
        )
        return dbc.Row(
            [
                dbc.Label(field_name, width=5, className="small"),
                dbc.Col(ctrl, width=7),
            ],
            className="mb-2",
        )

    # --- float (including Optional[float]) ----------------------------------
    if resolved is float:
        ctrl = dbc.Input(
            id=input_id,
            type="number",
            step=0.01,
            value=value if value is not None else "",
            size="sm",
        )
        return dbc.Row(
            [
                dbc.Label(field_name, width=5, className="small"),
                dbc.Col(ctrl, width=7),
            ],
            className="mb-2",
        )

    # --- str or fallback ----------------------------------------------------
    ctrl = dbc.Input(
        id=input_id,
        type="text",
        value=str(value) if value is not None else "",
        size="sm",
    )
    return dbc.Row(
        [
            dbc.Label(field_name, width=5, className="small"),
            dbc.Col(ctrl, width=7),
        ],
        className="mb-2",
    )


def render_param_editor(
    block_id: str | None,
    instance_id: str | None,
    params_class: type[Any] | None,
    current_values: dict[str, Any],
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

    rows: list[Any] = []
    for f in fields:
        default = f.default if f.default is not dataclasses.MISSING else None
        val = current_values.get(f.name, default)
        rows.append(_field_to_input(f, val, instance_id))

    return html.Div(
        [
            html.H6(block_id, className="mb-3"),
            dbc.Form(rows),
        ]
    )
