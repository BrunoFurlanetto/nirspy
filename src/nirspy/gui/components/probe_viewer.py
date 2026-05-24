"""Probe viewer component — 2D source/detector montage plot."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from dash import dcc, html


def render_probe_viewer(
    raw_info: Any = None,
    bads: list[str] | None = None,
) -> html.Div:
    """Render a 2D scatter plot of the fNIRS probe layout.

    Sources are red circles, detectors are blue squares.
    Bad (pruned) channels are shown in gray.

    Parameters
    ----------
    raw_info:
        An ``mne.Info`` instance, or *None*.
    bads:
        List of channel names marked as bad.

    Returns
    -------
    html.Div
        Plotly Graph or a graceful fallback message.
    """
    if raw_info is None:
        return html.Div(
            html.P(
                "No montage info available.",
                className="text-muted text-center py-4",
            ),
            id="probe-viewer-content",
        )

    # Try to extract montage positions
    try:
        montage = raw_info.get_montage()
        if montage is None:
            return html.Div(
                html.P(
                    "No montage info available.",
                    className="text-muted text-center py-4",
                ),
                id="probe-viewer-content",
            )

        positions = montage.get_positions()
        ch_pos = positions.get("ch_pos")
        if ch_pos is None or len(ch_pos) == 0:
            return html.Div(
                html.P(
                    "No montage info available.",
                    className="text-muted text-center py-4",
                ),
                id="probe-viewer-content",
            )
    except Exception:  # noqa: BLE001
        return html.Div(
            html.P(
                "No montage info available.",
                className="text-muted text-center py-4",
            ),
            id="probe-viewer-content",
        )

    bads_set = set(bads or [])
    ch_types: dict[str, str] = {}
    try:
        for ch_info in raw_info["chs"]:
            ch_types[ch_info["ch_name"]] = str(
                ch_info.get("kind", "")
            )
    except (KeyError, TypeError):
        pass

    # Classify channels as source-like or detector-like by name
    fig = go.Figure()
    src_x, src_y, src_names = [], [], []
    det_x, det_y, det_names = [], [], []
    bad_x, bad_y, bad_names = [], [], []

    for ch_name, pos in ch_pos.items():
        x_val, y_val = float(pos[0]), float(pos[1])
        if ch_name in bads_set:
            bad_x.append(x_val)
            bad_y.append(y_val)
            bad_names.append(ch_name)
        elif "S" in ch_name.upper().split("_")[0]:
            src_x.append(x_val)
            src_y.append(y_val)
            src_names.append(ch_name)
        else:
            det_x.append(x_val)
            det_y.append(y_val)
            det_names.append(ch_name)

    if src_x:
        fig.add_trace(
            go.Scatter(
                x=src_x,
                y=src_y,
                mode="markers",
                name="Sources",
                marker={
                    "color": "#d62728",
                    "size": 10,
                    "symbol": "circle",
                },
                text=src_names,
                hoverinfo="text",
            )
        )

    if det_x:
        fig.add_trace(
            go.Scatter(
                x=det_x,
                y=det_y,
                mode="markers",
                name="Detectors",
                marker={
                    "color": "#1f77b4",
                    "size": 10,
                    "symbol": "square",
                },
                text=det_names,
                hoverinfo="text",
            )
        )

    if bad_x:
        fig.add_trace(
            go.Scatter(
                x=bad_x,
                y=bad_y,
                mode="markers",
                name="Pruned",
                marker={
                    "color": "#999999",
                    "size": 10,
                    "symbol": "x",
                },
                text=bad_names,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Probe Layout",
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        height=400,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        template="plotly_white",
    )

    return html.Div(
        dcc.Graph(
            id="probe-viewer-graph",
            figure=fig,
        ),
        id="probe-viewer-content",
    )
