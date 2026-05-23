"""Generic per-block runtime dialog for interactive pipeline execution (T-027).

Renders a ``dbc.Modal`` containing the existing ``ParamEditor`` for the
current block.  The dialog is intentionally generic — specialized dialogs
for BlockAverage (T-028) and ManualChannelExclude (T-029) will override
this for their respective block types in future waves.

Design decisions:
- ``backdrop="static"`` and ``keyboard=False`` prevent accidental dismissal
  mid-run (same pattern as tutorial.py:109).
- Header shows block display-name + step counter ("Block N/M: <name>").
- Footer has three actions: Skip, Cancel, Run with these params.
- ``render_runtime_dialog`` is a pure render function (no callbacks here).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import dash_bootstrap_components as dbc

from nirspy.domain.block import BlockSpec
from nirspy.gui.components.param_editor import render_param_editor


def render_runtime_dialog(
    block_spec: BlockSpec,
    current_idx: int,
    total: int,
    current_params: dict[str, Any] | None = None,
) -> dbc.Modal:
    """Build the generic runtime dialog Modal for *block_spec*.

    Parameters
    ----------
    block_spec:
        The :class:`~nirspy.domain.block.BlockSpec` for the block about to
        be executed.
    current_idx:
        0-based index of the current block among enabled steps.
    total:
        Total number of enabled steps in the pipeline.
    current_params:
        Dict of current parameter values for the block.  When *None*, the
        block's dataclass defaults are used.

    Returns
    -------
    dbc.Modal
        Fully configured modal, initially open (``is_open=True``).
    """
    step_num = current_idx + 1  # 1-based for display
    header_text = (
        f"Block {step_num}/{total}: {block_spec.display_name}"
    )

    params_class = block_spec.params_class
    params_values: dict[str, Any] = {}
    if params_class is not None and dataclasses.is_dataclass(params_class):
        # Seed with dataclass defaults, then overlay current_params
        for f in dataclasses.fields(params_class):
            if f.default is not dataclasses.MISSING:
                params_values[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                params_values[f.name] = f.default_factory()
        if current_params:
            params_values.update(current_params)

    param_editor_content = render_param_editor(
        block_id=block_spec.block_id,
        instance_id=f"runtime-{block_spec.block_id}",
        params_class=params_class,
        current_values=params_values,
    )

    footer = dbc.ModalFooter(
        dbc.ButtonGroup(
            [
                dbc.Button(
                    "Skip",
                    id="runtime-skip-btn",
                    color="secondary",
                    outline=True,
                    size="sm",
                ),
                dbc.Button(
                    "Cancel run",
                    id="runtime-cancel-btn",
                    color="danger",
                    outline=True,
                    size="sm",
                ),
                dbc.Button(
                    "Run with these params",
                    id="runtime-advance-btn",
                    color="primary",
                    size="sm",
                ),
            ],
            size="sm",
        )
    )

    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(header_text),
                close_button=False,
            ),
            dbc.ModalBody(param_editor_content),
            footer,
        ],
        id="runtime-dialog-modal",
        is_open=True,
        backdrop="static",
        keyboard=False,
        size="lg",
    )
