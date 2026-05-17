"""Pipeline view component — vertical list of block cards.

Renders the current pipeline state as a sequence of
:func:`~nirspy.gui.components.block_card.render_block_card` components
with compatibility indicators between adjacent blocks.
"""

from __future__ import annotations

from typing import Any

from dash import html

from nirspy.blocks import registry
from nirspy.domain.block import BlockSpec
from nirspy.gui.components.block_card import render_block_card


def render_pipeline_view(
    pipeline_state: list[dict[str, Any]],
    selected_instance_id: str | None,
) -> html.Div:
    """Build the pipeline view from the current state.

    Parameters
    ----------
    pipeline_state:
        List of ``BlockStateDict`` dicts from ``dcc.Store("pipeline-state")``.
    selected_instance_id:
        ``instance_id`` of the currently selected block (or *None*).

    Returns
    -------
    html.Div
        Container with all block cards.
    """
    if not pipeline_state:
        return html.Div(
            html.P(
                "Add blocks from the catalog to build your pipeline.",
                className="text-muted text-center",
            )
        )

    cards: list[Any] = []
    n = len(pipeline_state)

    for idx, entry in enumerate(pipeline_state):
        block_id: str = entry["block_id"]
        instance_id: str = entry["instance_id"]
        enabled: bool = entry.get("enabled", True)

        # Resolve spec
        block_cls = registry.get(block_id)
        spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]

        # Previous block output type for compatibility indicator
        prev_output: str | None = None
        if idx > 0:
            prev_bid: str = pipeline_state[idx - 1]["block_id"]
            prev_cls = registry.get(prev_bid)
            prev_spec: BlockSpec = prev_cls.SPEC  # type: ignore[attr-defined]
            prev_output = prev_spec.output_type.value

        cards.append(
            render_block_card(
                block_id=block_id,
                instance_id=instance_id,
                display_name=spec.display_name,
                input_type=spec.input_type.value,
                output_type=spec.output_type.value,
                enabled=enabled,
                selected=(instance_id == selected_instance_id),
                is_first=(idx == 0),
                is_last=(idx == n - 1),
                prev_output_type=prev_output,
            )
        )

    return html.Div(cards)
