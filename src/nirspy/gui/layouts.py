"""Main layout for the NIRSPY Dash application.

Three-panel layout:
- Left sidebar: block catalog (populated from BlockRegistry)
- Center: pipeline view (vertical list of block cards)
- Right panel: parameter editor (auto-generated from dataclass fields)

State management uses ``dcc.Store`` for pipeline state (JSON-serializable).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from nirspy.blocks import registry
from nirspy.gui.components.block_catalog import render_block_catalog


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
                children=[render_block_catalog(registry)],
            ),
            html.Hr(),
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        "Save Pipeline",
                        id="btn-save-pipeline",
                        color="success",
                        size="sm",
                        className="me-1",
                    ),
                    dcc.Upload(
                        id="upload-pipeline",
                        children=dbc.Button(
                            "Load Pipeline",
                            color="info",
                            size="sm",
                        ),
                        accept=".yaml,.yml",
                    ),
                ],
                className="w-100",
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

    # Hidden stores for pipeline state + download component
    stores = html.Div(
        [
            dcc.Store(id="pipeline-state", data=[]),
            dcc.Store(id="selected-block", data=None),
            dcc.Download(id="download-pipeline"),
        ]
    )

    return dbc.Container(
        [navbar, body, stores],
        fluid=True,
        className="px-0",
    )
