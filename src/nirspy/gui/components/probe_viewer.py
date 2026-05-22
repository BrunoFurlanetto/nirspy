"""Probe viewer component — 2D source/detector montage plot."""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
from dash import dcc, html


def _draw_head_silhouette(fig: go.Figure, scale: float) -> None:
    """Add 2D top-view head shape traces to *fig* (D5).

    Draws an ellipse (scalp contour), triangle (nasion/nose at top),
    and ear markers (left/right). All traces are added below optode
    data by being inserted first.

    Parameters
    ----------
    fig:
        Plotly Figure to modify in-place.
    scale:
        Radius scale factor based on optode bounds.
    """
    if scale <= 0:
        scale = 0.05  # fallback default

    # Head ellipse (slightly taller than wide for top-view)
    n_pts = 100
    theta = [2 * math.pi * i / n_pts for i in range(n_pts + 1)]
    head_x = [scale * 1.1 * math.cos(t) for t in theta]
    head_y = [scale * 1.2 * math.sin(t) for t in theta]

    fig.add_trace(
        go.Scatter(
            x=head_x,
            y=head_y,
            mode="lines",
            name="Head",
            line={"color": "#cccccc", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Nose triangle (nasion indicator at top)
    nose_w = scale * 0.15
    nose_h = scale * 0.15
    nose_base_y = scale * 1.2
    fig.add_trace(
        go.Scatter(
            x=[-nose_w, 0, nose_w, -nose_w],
            y=[nose_base_y, nose_base_y + nose_h, nose_base_y, nose_base_y],
            mode="lines",
            name="Nasion",
            line={"color": "#cccccc", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Left ear
    ear_x_off = scale * 1.1
    ear_h = scale * 0.2
    ear_w = scale * 0.08
    fig.add_trace(
        go.Scatter(
            x=[-ear_x_off, -ear_x_off - ear_w, -ear_x_off],
            y=[-ear_h, 0, ear_h],
            mode="lines",
            name="Left ear",
            line={"color": "#cccccc", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Right ear
    fig.add_trace(
        go.Scatter(
            x=[ear_x_off, ear_x_off + ear_w, ear_x_off],
            y=[-ear_h, 0, ear_h],
            mode="lines",
            name="Right ear",
            line={"color": "#cccccc", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Inion marker (bottom text)
    fig.add_annotation(
        x=0,
        y=-scale * 1.3,
        text="Iz",
        showarrow=False,
        font={"size": 10, "color": "#999999"},
    )


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

    # Compute scale from optode bounds for head silhouette
    all_positions = list(ch_pos.values())
    if all_positions:
        pos_array = [[float(p[0]), float(p[1])] for p in all_positions]
        xs = [p[0] for p in pos_array]
        ys = [p[1] for p in pos_array]
        x_range = max(xs) - min(xs) if xs else 0.1
        y_range = max(ys) - min(ys) if ys else 0.1
        head_scale = max(x_range, y_range) * 0.6
    else:
        head_scale = 0.05

    _draw_head_silhouette(fig, head_scale)

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
