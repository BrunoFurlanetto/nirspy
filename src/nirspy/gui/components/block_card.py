"""Block card component — renders a single block in the pipeline view.

Displays the block name, input/output type badges, an enabled toggle,
and up/down/remove action buttons.  A highlighted border indicates
the currently selected block.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from nirspy.domain.data_types import DataType


def _io_compat_indicator(
    output_type: DataType | None,
    input_type: DataType | None,
) -> html.Span:
    """Small coloured dot between two blocks indicating type compatibility."""
    if output_type is None or input_type is None:
        return html.Span()
    compatible = (
        output_type is DataType.ANY
        or input_type is DataType.ANY
        or output_type == input_type
    )
    colour = "#28a745" if compatible else "#dc3545"
    return html.Span(
        "●",
        style={
            "color": colour,
            "fontSize": "12px",
            "display": "block",
            "textAlign": "center",
            "lineHeight": "16px",
        },
        title="Types compatible" if compatible else "Type mismatch",
    )


def render_block_card(
    *,
    block_id: str,
    instance_id: str,
    display_name: str,
    input_type: str,
    output_type: str,
    enabled: bool,
    selected: bool,
    is_first: bool,
    is_last: bool,
    prev_output_type: str | None = None,
) -> html.Div:
    """Render a single pipeline block card.

    Parameters
    ----------
    block_id:
        Registry block ID.
    instance_id:
        Unique session ID for this pipeline step.
    display_name:
        Human-readable block name.
    input_type, output_type:
        DataType *values* (strings).
    enabled:
        Whether the block is currently active.
    selected:
        Whether this card is the currently selected one.
    is_first, is_last:
        Position flags controlling up/down button disabled state.
    prev_output_type:
        Output type of the previous block (for compatibility indicator).
    """
    border_colour = "primary" if selected else ("secondary" if enabled else "light")
    opacity = "1" if enabled else "0.45"

    # Compatibility indicator above this card
    compat_dot = _io_compat_indicator(
        DataType(prev_output_type) if prev_output_type else None,
        DataType(input_type),
    )

    card = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span(
                                        display_name,
                                        className="fw-bold small",
                                    ),
                                    html.Div(
                                        [
                                            dbc.Badge(
                                                input_type,
                                                color="info",
                                                pill=True,
                                                className="me-1",
                                            ),
                                            html.Span(
                                                "→",
                                                className="text-muted small mx-1",
                                            ),
                                            dbc.Badge(
                                                output_type,
                                                color="info",
                                                pill=True,
                                            ),
                                        ],
                                        className="mt-1",
                                    ),
                                ],
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.Button(
                                        "▲",
                                        id={
                                            "type": "btn-up",
                                            "instance_id": instance_id,
                                        },
                                        size="sm",
                                        color="outline-secondary",
                                        className="me-1",
                                        disabled=is_first,
                                    ),
                                    dbc.Button(
                                        "▼",
                                        id={
                                            "type": "btn-down",
                                            "instance_id": instance_id,
                                        },
                                        size="sm",
                                        color="outline-secondary",
                                        className="me-1",
                                        disabled=is_last,
                                    ),
                                    dbc.Button(
                                        "✕",
                                        id={
                                            "type": "btn-remove",
                                            "instance_id": instance_id,
                                        },
                                        size="sm",
                                        color="outline-danger",
                                        className="me-1",
                                    ),
                                    dbc.Switch(
                                        id={
                                            "type": "switch-enable",
                                            "instance_id": instance_id,
                                        },
                                        value=enabled,
                                        className="d-inline-block ms-2",
                                    ),
                                ],
                                className="d-flex align-items-center justify-content-end",
                            ),
                            width=6,
                        ),
                    ],
                    align="center",
                ),
            ],
            className="p-2",
        ),
        id={"type": "block-card", "instance_id": instance_id},
        color=border_colour,
        outline=True,
        className="mb-1",
        style={"opacity": opacity, "cursor": "pointer"},
    )

    return html.Div([compat_dot, card])
