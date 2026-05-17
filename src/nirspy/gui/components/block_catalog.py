"""Block catalog component -- sidebar listing all available blocks.

Renders a clickable list of blocks from :data:`~nirspy.blocks.registry`.
Each entry shows the block display name, input/output type badges,
and a tooltip with scientific context (T-006 5D).
Clicking an entry triggers a callback that adds the block to the pipeline.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html

from nirspy.blocks.registry import BlockRegistry
from nirspy.domain.block import BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.gui.components.tooltips import tooltip_for


def _type_badge(dt: DataType, *, prefix: str = "") -> dbc.Badge:
    """Return a small coloured badge for a DataType."""
    color_map: dict[str, str] = {
        "raw": "primary",
        "raw_od": "info",
        "raw_haemo": "danger",
        "evoked": "warning",
        "epochs": "secondary",
        "dataframe": "dark",
        "any": "light",
    }
    color = color_map.get(dt.value, "secondary")
    label = f"{prefix}{dt.value}" if prefix else dt.value
    return dbc.Badge(label, color=color, className="me-1", pill=True)


def _catalog_item(block_id: str, spec: BlockSpec) -> dbc.ListGroupItem:
    """Render a single catalog entry as a clickable list-group item."""
    return dbc.ListGroupItem(
        [
            html.Div(spec.display_name, className="fw-bold small"),
            html.Div(
                [
                    _type_badge(spec.input_type, prefix=""),
                    html.Span(" -> ", className="text-muted small"),
                    _type_badge(spec.output_type, prefix=""),
                ],
                className="mt-1",
            ),
        ],
        id={"type": "catalog-item", "block_id": block_id},
        action=True,
        className="px-2 py-2",
    )


def render_block_catalog(registry: BlockRegistry) -> html.Div:
    """Build the full block catalog component.

    Parameters
    ----------
    registry:
        Populated :class:`~nirspy.blocks.registry.BlockRegistry`.

    Returns
    -------
    html.Div
        Div containing a ``dbc.ListGroup`` of catalog items
        with tooltips attached.
    """
    items: list[Any] = []
    tooltips: list[Any] = []
    for block_id in registry.list_blocks():
        block_cls = registry.get(block_id)
        spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]
        items.append(_catalog_item(block_id, spec))
        tt = tooltip_for(block_id)
        if tt is not None:
            tooltips.append(tt)

    return html.Div([dbc.ListGroup(items, flush=True)] + tooltips)
