"""Probe viewer component — 2D source/detector montage plot."""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
from dash import dcc, html

# Canonical head silhouette dimensions (viewport coordinates).
_HEAD_RX: float = 1.1
_HEAD_RY: float = 1.2
_OPTODE_FIT_RADIUS: float = 1.0

# 10-20 system reference positions (top-view, head-normalised).
_TEN_TWENTY: dict[str, tuple[float, float]] = {
    "Nz": (0.0, 1.2),
    "Fpz": (0.0, 0.95),
    "Fp1": (-0.30, 0.90),
    "Fp2": (0.30, 0.90),
    "F7": (-0.80, 0.55),
    "F3": (-0.40, 0.50),
    "Fz": (0.0, 0.50),
    "F4": (0.40, 0.50),
    "F8": (0.80, 0.55),
    "T7": (-1.00, 0.0),
    "C3": (-0.45, 0.0),
    "Cz": (0.0, 0.0),
    "C4": (0.45, 0.0),
    "T8": (1.00, 0.0),
    "P7": (-0.80, -0.55),
    "P3": (-0.40, -0.50),
    "Pz": (0.0, -0.50),
    "P4": (0.40, -0.50),
    "P8": (0.80, -0.55),
    "O1": (-0.30, -0.90),
    "O2": (0.30, -0.90),
    "Oz": (0.0, -0.95),
    "Iz": (0.0, -1.2),
}


def _pca_rotation_angle(centered_pts: list[tuple[float, float]]) -> float:
    """Angle (radians) of the principal axis vs. the x-axis."""
    if len(centered_pts) < 2:
        return 0.0
    sxx = sum(p[0] * p[0] for p in centered_pts)
    syy = sum(p[1] * p[1] for p in centered_pts)
    sxy = sum(p[0] * p[1] for p in centered_pts)
    return 0.5 * math.atan2(2.0 * sxy, sxx - syy)


def _normalize_xy(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Recenter, PCA-rotate (long axis -> x), uniform scale."""
    if not pts:
        return []
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    centered = [(p[0] - cx, p[1] - cy) for p in pts]
    theta = _pca_rotation_angle(centered)
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    rotated = [
        (p[0] * cos_t - p[1] * sin_t, p[0] * sin_t + p[1] * cos_t)
        for p in centered
    ]
    max_r = max(math.hypot(p[0], p[1]) for p in rotated)
    scale = 1.0 if max_r <= 0 else _OPTODE_FIT_RADIUS / max_r
    return [(p[0] * scale, p[1] * scale) for p in rotated]


def _draw_ten_twenty_overlay(fig: go.Figure) -> None:
    """Add a faint 10-20 reference layer."""
    xs = [v[0] for v in _TEN_TWENTY.values()]
    ys = [v[1] for v in _TEN_TWENTY.values()]
    labels = list(_TEN_TWENTY.keys())
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            name="10-20 reference",
            marker={
                "color": "rgba(120,120,120,0.35)",
                "size": 6,
                "symbol": "circle-open",
            },
            text=labels,
            textfont={"color": "rgba(80,80,80,0.55)", "size": 9},
            textposition="bottom center",
            hoverinfo="text",
            showlegend=True,
        )
    )


def _draw_head_silhouette(fig: go.Figure, scale: float = 1.0) -> None:
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
    rx = _HEAD_RX
    ry = _HEAD_RY

    n_pts = 100
    theta = [2 * math.pi * i / n_pts for i in range(n_pts + 1)]
    head_x = [rx * math.cos(t) for t in theta]
    head_y = [ry * math.sin(t) for t in theta]

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
    nose_w = 0.15
    nose_h = 0.15
    nose_base_y = ry
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
    ear_x_off = rx
    ear_h = 0.2
    ear_w = 0.08
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

    fig = go.Figure()

    # Normalise channel positions so the probe always lands inside the
    # canonical head silhouette regardless of the raw unit (m/mm/cm) or
    # any anatomical offset (e.g. PFC-only probes not centered at origin).
    ch_items = list(ch_pos.items())
    raw_pts = [
        (float(p[0]), float(p[1])) for _, p in ch_items
    ]
    norm_pts = _normalize_xy(raw_pts)

    _draw_head_silhouette(fig)
    _draw_ten_twenty_overlay(fig)

    src_x, src_y, src_names = [], [], []
    det_x, det_y, det_names = [], [], []
    bad_x, bad_y, bad_names = [], [], []

    for (ch_name, _pos), (x_val, y_val) in zip(ch_items, norm_pts, strict=False):
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
        xaxis={
            "visible": False,
            "range": [-(_HEAD_RX + 0.25), _HEAD_RX + 0.25],
        },
        yaxis={
            "visible": False,
            "scaleanchor": "x",
            "range": [-(_HEAD_RY + 0.3), _HEAD_RY + 0.3],
        },
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
