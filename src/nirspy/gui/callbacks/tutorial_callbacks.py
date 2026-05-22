"""Tutorial callbacks -- step navigation and pipeline auto-load.

Manages the tutorial overlay state via ``dcc.Store("tutorial-store")``.
When the tutorial starts, it automatically loads the *best-practices*
pipeline template so the user has a concrete pipeline to follow along.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

import yaml
from dash import (
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
    no_update,
)

from nirspy.blocks import registry
from nirspy.domain.block import BlockSpec
from nirspy.gui.components.tutorial import (
    TUTORIAL_STEPS,
    render_tutorial_modal,
)

# Re-export for app.py side-effect import
REGISTERED: bool = True

_TEMPLATE_NAME = "best-practices-block-design.yml"


def _load_template_pipeline() -> list[dict[str, Any]]:
    """Load the best-practices pipeline template as pipeline-state list.

    Returns
    -------
    list[dict[str, Any]]
        Pipeline state entries compatible with ``dcc.Store("pipeline-state")``.
    """
    # SEC-INFO-02: locate template via absolute paths only (no importlib
    # traversal with relative '../../' segments).
    import pathlib

    candidates = [
        pathlib.Path(__file__).resolve().parents[4]
        / "examples"
        / "pipelines"
        / _TEMPLATE_NAME,
        pathlib.Path.cwd() / "examples" / "pipelines" / _TEMPLATE_NAME,
    ]
    raw: str | None = None
    for p in candidates:
        if p.is_file():
            raw = p.read_text(encoding="utf-8")
            break
    if raw is None:
        return []

    data: dict[str, Any] = yaml.safe_load(raw)
    if not isinstance(data, dict):
        return []

    steps_raw: list[dict[str, Any]] = data.get("steps", [])
    raw_params_map: dict[str, Any] = data.get("params") or {}

    pipeline_state: list[dict[str, Any]] = []
    for step in steps_raw:
        block_id: str = step["block_id"]
        try:
            block_cls = registry.get(block_id)
        except KeyError:
            continue

        spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]
        raw_params: dict[str, Any] = raw_params_map.get(block_id, {})
        if (
            not raw_params
            and spec.params_class is not None
            and dataclasses.is_dataclass(spec.params_class)
        ):
            try:
                default_obj = spec.params_class()
                raw_params = dataclasses.asdict(default_obj)
            except TypeError:
                raw_params = {}

        pipeline_state.append(
            {
                "block_id": block_id,
                "instance_id": str(uuid.uuid4()),
                "params": raw_params,
                "enabled": step.get("enabled", True),
            }
        )

    return pipeline_state


# ── Start tutorial ──────────────────────────────────────────────────
@callback(
    Output("tutorial-store", "data", allow_duplicate=True),
    Output("pipeline-state", "data", allow_duplicate=True),
    Input("btn-start-tutorial", "n_clicks"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def start_tutorial(
    n_clicks: int | None,
    current_pipeline: list[dict[str, Any]],
) -> tuple[dict[str, Any], Any]:
    """Activate the tutorial and auto-load the demo pipeline."""
    if not n_clicks:
        return no_update, no_update  # type: ignore[return-value]

    store = {"active": True, "step": 0}
    template = _load_template_pipeline()
    pipeline = template if template else no_update
    return store, pipeline


# ── Navigate: next / prev / skip / finish ───────────────────────────
@callback(
    Output("tutorial-store", "data", allow_duplicate=True),
    Input("tutorial-next", "n_clicks"),
    Input("tutorial-prev", "n_clicks"),
    Input("tutorial-skip", "n_clicks"),
    Input("tutorial-finish", "n_clicks"),
    State("tutorial-store", "data"),
    prevent_initial_call=True,
)
def navigate_tutorial(
    next_clicks: int | None,
    prev_clicks: int | None,
    skip_clicks: int | None,
    finish_clicks: int | None,
    store: dict[str, Any],
) -> dict[str, Any]:
    """Advance, go back, skip, or finish the tutorial."""
    if not store or not store.get("active"):
        return no_update  # type: ignore[return-value]

    triggered = ctx.triggered_id
    step: int = store.get("step", 0)
    total = len(TUTORIAL_STEPS)

    if triggered == "tutorial-next":
        step = min(step + 1, total - 1)
    elif triggered == "tutorial-prev":
        step = max(step - 1, 0)
    elif triggered in ("tutorial-skip", "tutorial-finish"):
        return {"active": False, "step": 0}

    return {"active": True, "step": step}


# ── Render modal + highlight ────────────────────────────────────────
@callback(
    Output("tutorial-modal-container", "children"),
    Output("tutorial-highlight-target", "data"),
    Input("tutorial-store", "data"),
)
def render_tutorial(
    store: dict[str, Any] | None,
) -> tuple[list[Any], str]:
    """Update modal content and highlight target based on store state."""
    if not store or not store.get("active"):
        return [], ""

    step: int = store.get("step", 0)
    modal = render_tutorial_modal(step)
    target_id = TUTORIAL_STEPS[step].target_id
    return [modal], target_id


# ── Client-side highlight via className manipulation ────────────────
# Dash clientside callback to toggle CSS highlight class on the target
# element in the browser (no server round-trip).
clientside_callback(
    """
    function(targetId) {
        // Remove previous highlights
        var prev = document.querySelectorAll('.tutorial-highlight');
        for (var i = 0; i < prev.length; i++) {
            prev[i].classList.remove('tutorial-highlight');
        }
        // Add highlight to current target
        if (targetId) {
            var el = document.getElementById(targetId);
            if (el) {
                el.classList.add('tutorial-highlight');
                el.scrollIntoView({behavior: 'smooth', block: 'center'});
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("tutorial-highlight-target", "id"),
    Input("tutorial-highlight-target", "data"),
)
