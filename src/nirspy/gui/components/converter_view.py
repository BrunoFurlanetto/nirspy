"""Converter view component -- upload, convert, and download .nirs/.snirf files.

Provides the UI for the "Convert" tab in the main layout.  Users can
upload a ``.nirs`` or ``.snirf`` file, choose the conversion direction,
optionally strip PII, and download the converted file.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def render_converter_tab() -> html.Div:
    """Build the converter tab content.

    Returns
    -------
    html.Div
        Self-contained converter UI with upload, controls, status,
        and download components.
    """
    upload_area = dcc.Upload(
        id="converter-upload",
        children=dbc.Card(
            dbc.CardBody(
                [
                    html.I(
                        className="bi bi-cloud-arrow-up fs-1 "
                        "text-primary"
                    ),
                    html.P(
                        "Drag & drop or click to upload a "
                        ".nirs or .snirf file",
                        className="text-muted mt-2 mb-0",
                    ),
                ],
                className="text-center py-4",
            ),
            className="border-dashed",
        ),
        accept=".nirs,.snirf,.txt",
        multiple=False,
        className="mb-3",
    )

    direction_toggle = dbc.RadioItems(
        id="converter-direction",
        options=[
            {
                "label": ".nirs → .snirf",
                "value": "nirs_to_snirf",
            },
            {
                "label": ".snirf → .nirs",
                "value": "snirf_to_nirs",
            },
            {
                "label": "Oxysoft .txt → .snirf",
                "value": "oxysoft_txt_to_snirf",
            },
        ],
        value="nirs_to_snirf",
        inline=True,
        className="mb-3",
    )

    strip_pii_checkbox = dbc.Checkbox(
        id="converter-strip-pii",
        label="Strip PII (remove subject identifiers)",
        value=False,
        className="mb-3",
    )

    convert_button = dbc.Button(
        "Convert",
        id="converter-btn-convert",
        color="primary",
        disabled=True,
        className="w-100 mb-3",
    )

    status_area = html.Div(
        id="converter-status",
        children=[],
    )

    download = dcc.Download(id="converter-download")

    filename_display = html.Div(
        id="converter-filename",
        children=[],
        className="mb-3",
    )

    return html.Div(
        [
            html.H5("File Converter", className="mb-3"),
            html.P(
                "Convert between .nirs (HOMER2/3), .snirf (SNIRF 1.1) and "
                "Oxysoft .txt exports. Direct .oxy3 → .snirf is planned for "
                "a future release pending an open-source parser.",
                className="text-muted small",
            ),
            upload_area,
            filename_display,
            html.Hr(),
            html.Label(
                "Conversion direction",
                className="fw-bold small",
            ),
            direction_toggle,
            strip_pii_checkbox,
            convert_button,
            status_area,
            download,
        ],
        className="p-3",
    )
