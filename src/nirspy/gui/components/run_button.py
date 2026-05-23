"""Run button + progress bar component for pipeline execution."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def render_run_button() -> html.Div:
    """Render the Run Pipeline button with progress bar and alerts.

    Returns
    -------
    html.Div
        Container with button, progress bar, success/error alerts,
        and an upload component for the input SNIRF file.
    """
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Upload(
                            id="upload-input-file",
                            children=dbc.Button(
                                "Select SNIRF file",
                                color="outline-secondary",
                                size="sm",
                                className="w-100",
                            ),
                            accept=".snirf",
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Run Pipeline",
                            id="run-button",
                            color="success",
                            size="sm",
                            className="w-100",
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Run Interactive",
                            id="run-interactive-btn",
                            color="info",
                            size="sm",
                            className="w-100",
                        ),
                        width=4,
                    ),
                ],
                className="mb-2",
            ),
            html.Div(
                id="input-file-label",
                children=html.Small(
                    "No file selected",
                    className="text-muted",
                ),
            ),
            dbc.Progress(
                id="run-progress",
                value=0,
                max=100,
                striped=True,
                animated=True,
                className="mt-2",
                style={"display": "none"},
            ),
            dbc.Alert(
                id="run-error",
                color="danger",
                dismissable=True,
                is_open=False,
                className="mt-2",
            ),
            dbc.Alert(
                id="run-success",
                color="success",
                dismissable=True,
                is_open=False,
                className="mt-2",
            ),
            # Interactive run alerts (shared with runtime_callbacks)
            dbc.Alert(
                id="run-interactive-error",
                color="danger",
                dismissable=True,
                is_open=False,
                className="mt-2",
            ),
            dbc.Alert(
                id="run-interactive-warning",
                color="warning",
                dismissable=True,
                is_open=False,
                className="mt-2",
            ),
        ],
        className="mb-3",
    )
