"""Main layout for the NIRSPY Dash application.

Three-panel layout:
- Left sidebar: block catalog
- Center: pipeline view (vertical list of blocks)
- Right panel: parameter editor (placeholder for 5B)

State management uses ``dcc.Store`` for pipeline state (JSON-serializable).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_layout() -> dbc.Container:
    """Build the top-level layout structure.

    Returns
    -------
    dbc.Container
        Bootstrap fluid container with navbar + three-column body.
    """
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("NIRSPY", className="ms-2 fw-bold"),
                dbc.NavbarToggler(id="navbar-toggler"),
                html.Span(
                    "fNIRS Pipeline Builder",
                    className="text-muted small ms-3 d-none d-md-inline",
                ),
            ],
            fluid=True,
        ),
        color="primary",
        dark=True,
        className="mb-3",
    )

    sidebar = dbc.Col(
        [
            html.H5("Block Catalog", className="text-center mb-3"),
            html.Hr(),
            html.Div(
                id="block-catalog",
                children=[
                    html.P(
                        "Available blocks will appear here.",
                        className="text-muted small",
                    )
                ],
            ),
        ],
        width=3,
        className="bg-light p-3 border-end",
        style={"minHeight": "80vh"},
    )

    center = dbc.Col(
        [
            html.H5("Pipeline", className="text-center mb-3"),
            html.Hr(),
            html.Div(
                id="pipeline-view",
                children=[
                    html.P(
                        "Add blocks from the catalog to build your pipeline.",
                        className="text-muted text-center",
                    )
                ],
            ),
        ],
        width=6,
        className="p-3",
    )

    right_panel = dbc.Col(
        [
            html.H5("Parameters", className="text-center mb-3"),
            html.Hr(),
            html.Div(
                id="param-editor",
                children=[
                    html.P(
                        "Select a block to edit its parameters.",
                        className="text-muted small",
                    )
                ],
            ),
        ],
        width=3,
        className="bg-light p-3 border-start",
        style={"minHeight": "80vh"},
    )

    body = dbc.Row(
        [sidebar, center, right_panel],
        className="g-0",
    )

    # Hidden stores for pipeline state (5B will populate these)
    stores = html.Div(
        [
            dcc.Store(id="pipeline-state", data=[]),
            dcc.Store(id="selected-block", data=None),
        ]
    )

    return dbc.Container(
        [navbar, body, stores],
        fluid=True,
        className="px-0",
    )
