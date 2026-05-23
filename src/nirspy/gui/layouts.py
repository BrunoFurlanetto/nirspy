"""Main layout for the NIRSPY Dash application.

Three-panel layout:
- Left sidebar: block catalog (populated from BlockRegistry)
- Center: pipeline view (vertical list of block cards)
- Right panel: parameter editor (auto-generated from dataclass fields)

Below the builder: visualization tabs (raw data, probe, QC, HRF)
and a Convert tab for .nirs/.snirf file conversion.

State management uses ``dcc.Store`` for pipeline state (JSON-serializable).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from nirspy.blocks import registry
from nirspy.gui.components.block_catalog import render_block_catalog
from nirspy.gui.components.condition_selector import render_condition_selector
from nirspy.gui.components.converter_view import render_converter_tab
from nirspy.gui.components.run_button import render_run_button


def create_layout() -> dbc.Container:
    """Build the top-level layout structure.

    Returns
    -------
    dbc.Container
        Bootstrap fluid container with navbar + three-column body
        + visualization tabs.
    """
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("NIRSPY", className="ms-2 fw-bold"),
                dbc.NavbarToggler(id="navbar-toggler"),
                html.Span(
                    "fNIRS Pipeline Builder",
                    className=(
                        "text-muted small ms-3 d-none d-md-inline"
                    ),
                ),
                dbc.Button(
                    "Tutorial",
                    id="btn-start-tutorial",
                    color="light",
                    size="sm",
                    className="ms-auto",
                    outline=True,
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
            html.H5(
                "Block Catalog", className="text-center mb-3"
            ),
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
            html.H5(
                "Pipeline", className="text-center mb-3"
            ),
            html.Hr(),
            html.Div(
                id="pipeline-view",
                children=[
                    html.P(
                        "Add blocks from the catalog to "
                        "build your pipeline.",
                        className="text-muted text-center",
                    )
                ],
            ),
            html.Hr(),
            render_run_button(),
        ],
        width=6,
        className="p-3",
    )

    right_panel = dbc.Col(
        [
            html.H5(
                "Parameters", className="text-center mb-3"
            ),
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

    # Visualization area below the builder
    viz_tabs = dbc.Tabs(
        [
            dbc.Tab(
                html.Div(
                    id="raw-data-plot-container",
                    children=html.P(
                        "No raw data available. "
                        "Run the pipeline first.",
                        className="text-muted text-center py-4",
                    ),
                ),
                label="Raw Data",
                tab_id="tab-raw",
            ),
            dbc.Tab(
                html.Div(
                    id="probe-viewer-container",
                    children=html.P(
                        "No montage info available.",
                        className="text-muted text-center py-4",
                    ),
                ),
                label="Probe",
                tab_id="tab-probe",
            ),
            dbc.Tab(
                html.Div(
                    id="qc-dashboard-container",
                    children=html.P(
                        "No QC data available.",
                        className="text-muted text-center py-4",
                    ),
                ),
                label="QC",
                tab_id="tab-qc",
            ),
            dbc.Tab(
                html.Div(
                    [
                        html.Div(
                            id="condition-selector-container",
                            children=render_condition_selector(None),
                        ),
                        # Discard-region controls — static; not re-rendered
                        # by the HRF callback so user state is preserved.
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Switch(
                                        id="hrf-discard-toggle",
                                        label=(
                                            "Show discard region "
                                            "(visual only — does not "
                                            "affect computation)"
                                        ),
                                        value=False,
                                        className="mt-2",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("t min (s)"),
                                            dbc.Input(
                                                id="hrf-discard-tmin",
                                                type="number",
                                                value=-5,
                                                step=0.5,
                                                style={"maxWidth": "90px"},
                                            ),
                                            dbc.InputGroupText("t max (s)"),
                                            dbc.Input(
                                                id="hrf-discard-tmax",
                                                type="number",
                                                value=5,
                                                step=0.5,
                                                style={"maxWidth": "90px"},
                                            ),
                                        ],
                                        size="sm",
                                    ),
                                    width="auto",
                                    id="hrf-discard-trange-col",
                                ),
                            ],
                            className="mb-2 align-items-center",
                            id="hrf-discard-controls",
                        ),
                        html.Div(
                            id="hrf-plot-container",
                            children=html.P(
                                "No HRF data available.",
                                className=(
                                    "text-muted text-center py-4"
                                ),
                            ),
                        ),
                    ],
                ),
                label="HRF",
                tab_id="tab-hrf",
            ),
            dbc.Tab(
                render_converter_tab(),
                label="Convert",
                tab_id="tab-convert",
            ),
        ],
        id="viz-tabs",
        active_tab="tab-raw",
        className="mt-3",
    )

    # Hidden stores for pipeline state + download + results + tutorial
    stores = html.Div(
        [
            dcc.Store(id="pipeline-state", data=[]),
            dcc.Store(id="selected-block", data=None),
            dcc.Store(id="run-results", data=None),
            dcc.Store(id="input-file-path", data=None),
            dcc.Store(
                id="tutorial-store",
                data={"active": False, "step": 0},
            ),
            dcc.Store(id="tutorial-highlight-target", data=""),
            dcc.Download(id="download-pipeline"),
            # Interactive run state store
            dcc.Store(
                id="interactive-exec-state",
                data={
                    "runner_id": "",
                    "current_idx": -1,
                    "status": "idle",
                },
            ),
        ]
    )

    # Tutorial modal container (populated dynamically by callbacks)
    tutorial_container = html.Div(id="tutorial-modal-container")

    # Runtime dialog container (populated dynamically by runtime_callbacks)
    runtime_dialog_container = html.Div(
        id="runtime-dialog-container",
        children=[],
    )

    return dbc.Container(
        [navbar, body, viz_tabs, stores, tutorial_container,
         runtime_dialog_container],
        fluid=True,
        className="px-0",
    )
