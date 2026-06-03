"""Pipeline callbacks -- add, remove, reorder, select, toggle enable.

All mutation goes through ``dcc.Store("pipeline-state")`` and
``dcc.Store("selected-block")``.  Callbacks are stateless; the store
is the single source of truth.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from typing import Any

from dash import ALL, Input, Output, State, callback, ctx, no_update

logger = logging.getLogger(__name__)

from nirspy.blocks import registry
from nirspy.domain.block import BlockSpec
from nirspy.gui.components.param_editor import render_param_editor
from nirspy.gui.components.pipeline_view import render_pipeline_view

# ---------------------------------------------------------------------------
# Condition-config modal callbacks (T-042j, T-042k)
# ---------------------------------------------------------------------------


@callback(
    Output("global-conditions-store", "data"),
    Output("condition-config-modal", "is_open", allow_duplicate=True),
    Output("condition-config-warning", "children", allow_duplicate=True),
    Output("condition-config-warning", "style", allow_duplicate=True),
    Output("condition-config-state", "data", allow_duplicate=True),
    Input("condition-config-apply-btn", "n_clicks"),
    State("condition-config-state", "data"),
    State("global-conditions-store", "data"),
    prevent_initial_call=True,
)
def apply_condition_config(
    n_clicks: int | None,
    state: dict[str, Any] | None,
    prev_gc: dict[str, Any] | None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Build GlobalConditions from modal state and persist to store.

    Validates the modal state, constructs a :class:`GlobalConditions` object,
    serialises it via ``global_conditions_to_dict``, and writes it to the
    ``global-conditions-store``.  Closes the modal on success; shows an
    inline warning on validation error.

    With ``debounce=True`` on all number inputs, values are sent to the server
    on blur (field loses focus) or Enter — before Apply is clicked. This means
    ``condition-config-state`` always holds the correct values by the time Apply
    fires, so no clientside snapshot is needed.

    Duration fallback order:
    1. ``state`` — last value synced from the debounced inputs (authoritative).
    2. ``prev_gc`` store — previously applied value for this condition.
    """
    if not n_clicks or not state:
        return no_update, no_update, no_update, no_update, no_update

    from nirspy.domain.conditions import (  # noqa: I001
        ConditionConfig,
        ConditionGroup as DomainConditionGroup,
        GlobalConditions,
        global_conditions_to_dict,
    )

    raw_conditions: list[dict[str, Any]] = state.get("conditions", [])
    raw_groups: list[dict[str, Any]] = state.get("groups", [])

    # Build lookup of previous GC durations as last-resort fallback
    _prev_dur: dict[str, float] = {}
    if prev_gc:
        for _c in prev_gc.get("conditions") or []:
            _prev_dur[_c.get("original_name", "")] = float(_c.get("duration", 1.0))

    # Build ConditionConfig list
    condition_configs: list[ConditionConfig] = []
    for raw_idx, cond in enumerate(raw_conditions):
        orig = cond.get("original_name") or cond.get("name", "")
        name = (cond.get("name") or orig).strip()
        if not name:
            continue
        occs = cond.get("occurrences", [])
        selected_occs = [o["idx"] for o in occs if o.get("selected", True)]
        included: tuple[int, ...] | None = (
            tuple(selected_occs) if len(selected_occs) < len(occs) else None
        )
        try:
            state_dur = cond.get("duration")
            logger.debug(
                "[apply_condition_config] cond=%r idx=%d state_dur=%r prev_gc_dur=%r",
                orig,
                raw_idx,
                state_dur,
                _prev_dur.get(orig),
            )
            if state_dur is not None:
                try:
                    dur = float(state_dur)
                except (TypeError, ValueError):
                    dur = _prev_dur.get(orig, 1.0)
            else:
                dur = _prev_dur.get(orig, 1.0)
            logger.debug(
                "[apply_condition_config] cond=%r → dur_final=%r",
                orig,
                dur,
            )
            tmin = float(cond.get("tmin", -2.0))
            tmax = float(cond.get("tmax", 18.0))
            btmin = float(cond.get("baseline_tmin", -2.0))
            btmax = float(cond.get("baseline_tmax", 0.0))
        except (TypeError, ValueError):
            continue
        try:
            condition_configs.append(
                ConditionConfig(
                    name=name,
                    original_name=orig,
                    included_occurrences=included,
                    duration=dur,
                    tmin=tmin,
                    tmax=tmax,
                    baseline_tmin=btmin,
                    baseline_tmax=btmax,
                )
            )
        except ValueError as exc:
            return (
                no_update,
                True,  # keep modal open
                f"Validation error in condition '{name}': {exc}",
                {"display": "block", "color": "red"},
                no_update,
            )

    if not condition_configs:
        return (
            no_update,
            True,
            "At least one condition must be configured.",
            {"display": "block", "color": "red"},
            no_update,
        )

    # Build ConditionGroup list (optional)
    domain_groups: list[DomainConditionGroup] = []
    for grp in raw_groups:
        label = (grp.get("label") or "").strip()
        if not label:
            continue
        cond_names = [c for c in (grp.get("conditions") or []) if c]
        if not cond_names:
            continue
        try:
            domain_groups.append(
                DomainConditionGroup(
                    label=label,
                    conditions=tuple(cond_names),
                    tmin=float(grp.get("tmin", -2.0)),
                    tmax=float(grp.get("tmax", 18.0)),
                    baseline_tmin=float(grp.get("baseline_tmin", -2.0)),
                    baseline_tmax=float(grp.get("baseline_tmax", 0.0)),
                )
            )
        except ValueError as exc:
            return (
                no_update,
                True,
                f"Validation error in group '{label}': {exc}",
                {"display": "block", "color": "red"},
                no_update,
            )

    try:
        gc = GlobalConditions(
            conditions=tuple(condition_configs),
            groups=tuple(domain_groups) if domain_groups else None,
        )
    except ValueError as exc:
        return (
            no_update,
            True,
            f"Validation error: {exc}",
            {"display": "block", "color": "red"},
            no_update,
        )

    serialised = global_conditions_to_dict(gc)
    logger.debug(
        "[apply_condition_config] SUCCESS — serialised durations: %s",
        {c["original_name"]: c["duration"] for c in serialised.get("conditions", [])},
    )

    # Build updated condition-config-state to keep state in sync with store
    occ_lookup: dict[str, list[Any]] = {
        c.get("original_name", ""): c.get("occurrences", [])
        for c in (state or {}).get("conditions", [])
    }
    synced_conditions = [
        {
            "name": cc.name,
            "original_name": cc.original_name,
            "duration": cc.duration,
            "tmin": cc.tmin,
            "tmax": cc.tmax,
            "baseline_tmin": cc.baseline_tmin,
            "baseline_tmax": cc.baseline_tmax,
            "occurrences": occ_lookup.get(cc.original_name, []),
        }
        for cc in condition_configs
    ]
    synced_state: dict[str, Any] = {
        **(state or {}),
        "conditions": synced_conditions,
        "_open": False,
    }
    return serialised, False, "", {"display": "none"}, synced_state


@callback(
    Output("condition-config-modal", "is_open", allow_duplicate=True),
    Input("condition-config-cancel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_condition_config(n_clicks: int | None) -> Any:
    """Close the condition config modal without persisting changes."""
    if not n_clicks:
        return no_update
    return False

# Re-export for app.py side-effect import
REGISTERED: bool = True


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input({"type": "catalog-item", "block_id": ALL}, "n_clicks"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def add_block(n_clicks: list[int | None], pipeline_state: list[dict[str, Any]]) -> Any:
    """Append a block to the pipeline when a catalog item is clicked."""
    if not ctx.triggered_id or all(c is None for c in n_clicks):
        return no_update

    block_id: str = ctx.triggered_id["block_id"]

    block_cls = registry.get(block_id)
    spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]
    default_params: dict[str, Any] = {}
    if spec.params_class is not None and dataclasses.is_dataclass(spec.params_class):
        try:
            default_obj = spec.params_class()
            default_params = dataclasses.asdict(default_obj)
        except TypeError:
            for f in dataclasses.fields(spec.params_class):
                if f.default is not dataclasses.MISSING:
                    default_params[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    default_params[f.name] = f.default_factory()
                else:
                    default_params[f.name] = None

    new_entry: dict[str, Any] = {
        "block_id": block_id,
        "instance_id": str(uuid.uuid4()),
        "params": default_params,
        "enabled": True,
    }

    state = list(pipeline_state) if pipeline_state else []
    state.append(new_entry)
    return state


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Output("selected-block", "data", allow_duplicate=True),
    Input({"type": "btn-remove", "instance_id": ALL}, "n_clicks"),
    State("pipeline-state", "data"),
    State("selected-block", "data"),
    prevent_initial_call=True,
)
def remove_block(
    n_clicks: list[int | None],
    pipeline_state: list[dict[str, Any]],
    selected: str | None,
) -> tuple[Any, Any]:
    """Remove a block from the pipeline."""
    if not ctx.triggered_id or all(c is None for c in n_clicks):
        return no_update, no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    state = [e for e in pipeline_state if e["instance_id"] != instance_id]
    new_selected = None if selected == instance_id else selected
    return state, new_selected


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input({"type": "btn-up", "instance_id": ALL}, "n_clicks"),
    Input({"type": "btn-down", "instance_id": ALL}, "n_clicks"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def reorder_blocks(
    up_clicks: list[int | None],
    down_clicks: list[int | None],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Move a block up or down in the pipeline."""
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    direction: str = ctx.triggered_id["type"]

    state = list(pipeline_state)
    idx = next((i for i, e in enumerate(state) if e["instance_id"] == instance_id), None)
    if idx is None:
        return no_update

    if direction == "btn-up" and idx > 0:
        state[idx], state[idx - 1] = state[idx - 1], state[idx]
    elif direction == "btn-down" and idx < len(state) - 1:
        state[idx], state[idx + 1] = state[idx + 1], state[idx]
    else:
        return no_update

    return state


@callback(
    Output("selected-block", "data", allow_duplicate=True),
    Input({"type": "block-card", "instance_id": ALL}, "n_clicks"),
    State("selected-block", "data"),
    prevent_initial_call=True,
)
def select_block(
    n_clicks: list[int | None],
    current_selected: str | None,
) -> Any:
    """Set the selected block when a card is clicked."""
    if not ctx.triggered_id:
        return no_update
    # Identify the n_clicks for the specific card that triggered. If it is
    # zero/None we are on the initial render fire — don't change selection.
    triggered_index = next(
        (
            i
            for i, inp in enumerate(ctx.inputs_list[0])
            if inp.get("id") == ctx.triggered_id
        ),
        None,
    )
    if triggered_index is None:
        return no_update
    n = n_clicks[triggered_index]
    if not n:
        return no_update
    instance_id: str = ctx.triggered_id["instance_id"]
    if instance_id == current_selected:
        return None
    return instance_id


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input({"type": "switch-enable", "instance_id": ALL}, "value"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def toggle_enable(
    values: list[bool],
    pipeline_state: list[dict[str, Any]],
) -> Any:
    """Update enabled state of a block."""
    if not ctx.triggered_id:
        return no_update

    instance_id: str = ctx.triggered_id["instance_id"]
    state = list(pipeline_state)

    triggered_value = ctx.triggered[0]["value"] if ctx.triggered else None
    if triggered_value is not None:
        for entry in state:
            if entry["instance_id"] == instance_id:
                entry["enabled"] = bool(triggered_value)
                break

    return state


@callback(
    Output("pipeline-view", "children"),
    Input("pipeline-state", "data"),
    Input("selected-block", "data"),
)
def render_pipeline(
    pipeline_state: list[dict[str, Any]] | None,
    selected: str | None,
) -> Any:
    """Re-render the pipeline view when state changes."""
    state = pipeline_state or []
    return render_pipeline_view(state, selected)


@callback(
    Output("param-editor", "children"),
    Input("selected-block", "data"),
    Input("pipeline-state", "data"),
    Input("global-conditions-store", "data"),
)
def render_params(
    selected: str | None,
    pipeline_state: list[dict[str, Any]] | None,
    global_conditions: dict[str, Any] | None,
) -> Any:
    """Re-render the parameter editor when the selection or state changes."""
    gc_active: bool = bool(global_conditions)

    if not selected or not pipeline_state:
        return render_param_editor(
            block_id=None,
            instance_id=None,
            params_class=None,
            current_values={},
        )

    entry = next((e for e in pipeline_state if e["instance_id"] == selected), None)
    if entry is None:
        return render_param_editor(
            block_id=None,
            instance_id=None,
            params_class=None,
            current_values={},
        )

    block_id: str = entry["block_id"]
    block_cls = registry.get(block_id)
    spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]

    # For blocks with per-condition widget (block_average, epochs_extraction),
    # harvest available condition names + SNIRF path from any LoadSnirf step in
    # the pipeline so the editor dropdown and timeline (T-030) are pre-populated
    # without requiring a pipeline run first.
    _BLOCKS_WITH_CONDITION_WIDGET = ("block_average", "epochs_extraction")
    available_conditions: list[str] | None = None
    snirf_path: str | None = None
    if block_id in _BLOCKS_WITH_CONDITION_WIDGET:
        from nirspy.gui.components.condition_windows_editor import (
            read_snirf_condition_names,
        )

        for step in pipeline_state:
            if step.get("block_id") == "load_snirf":
                raw_path = step.get("params", {}).get("path")
                if raw_path:
                    snirf_path = str(raw_path)
                    names = read_snirf_condition_names(snirf_path)
                    if names:
                        available_conditions = names
                    break

    return render_param_editor(
        block_id=block_id,
        instance_id=entry["instance_id"],
        params_class=spec.params_class,
        current_values=entry.get("params", {}),
        available_conditions=available_conditions,
        snirf_path=snirf_path,
        global_conditions_active=gc_active,
    )


@callback(
    Output("condition-config-state", "data", allow_duplicate=True),
    Output("condition-modal-open-trigger", "data", allow_duplicate=True),
    Input("btn-edit-conditions", "n_clicks"),
    State("condition-config-state", "data"),
    State("global-conditions-store", "data"),
    prevent_initial_call=True,
)
def open_condition_modal_from_button(
    n_clicks: int | None,
    state: dict[str, Any] | None,
    global_conditions: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Re-open the condition config modal, restoring last-applied values.

    Writes the restored state to ``condition-config-state`` and fires the
    dedicated ``condition-modal-open-trigger`` store.  ``_populate_modal``
    listens on the trigger rather than on the state store directly, so it is
    not serialised with ``_sync_condition_inputs`` by Dash — eliminating the
    keystroke race condition.
    """
    if not n_clicks or not global_conditions:
        return no_update, no_update

    logger.debug(
        "[open_condition_modal_from_button] gc_store durations: %s",
        {
            c.get("original_name", ""): c.get("duration")
            for c in (global_conditions.get("conditions") or [])
        },
    )
    logger.debug(
        "[open_condition_modal_from_button] state durations before rebuild: %s",
        {
            c.get("original_name", ""): c.get("duration")
            for c in ((state or {}).get("conditions") or [])
        },
    )

    # occurrences only live in condition-config-state (not serialised to global store)
    occ_by_orig: dict[str, list[Any]] = {
        c.get("original_name", ""): c.get("occurrences", [])
        for c in (state or {}).get("conditions", [])
    }

    # Rebuild conditions from global-conditions-store as source of truth
    gc_conditions: list[dict[str, Any]] = global_conditions.get("conditions") or []
    restored: list[dict[str, Any]] = []
    for gc_cond in gc_conditions:
        orig = gc_cond.get("original_name", "")
        restored.append({
            "name": gc_cond["name"],
            "original_name": orig,
            "duration": gc_cond["duration"],
            "tmin": gc_cond["tmin"],
            "tmax": gc_cond["tmax"],
            "baseline_tmin": gc_cond["baseline_tmin"],
            "baseline_tmax": gc_cond["baseline_tmax"],
            "occurrences": occ_by_orig.get(orig, []),
        })

    logger.debug(
        "[open_condition_modal_from_button] restored durations: %s",
        {c["original_name"]: c["duration"] for c in restored},
    )

    new_state: dict[str, Any] = dict(state) if state else {}
    new_state["conditions"] = restored

    gc_groups = global_conditions.get("groups")
    if gc_groups is not None:
        new_state["groups"] = gc_groups

    # Fire the dedicated open trigger instead of setting _open=True in state.
    # This keeps _populate_modal's Input separate from condition-config-state,
    # so Dash does not serialise it with _sync_condition_inputs.
    open_trigger = {"ts": n_clicks}
    return new_state, open_trigger


@callback(
    Output("btn-edit-conditions", "disabled"),
    Input("global-conditions-store", "data"),
)
def toggle_edit_conditions_btn(global_conditions: dict[str, Any] | None) -> bool:
    """Disable the Edit Conditions button when no global conditions are loaded."""
    return not bool(global_conditions)
