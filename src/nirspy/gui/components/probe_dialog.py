"""Probe interactive dialog for ManualChannelExclude (T-029).

Opened automatically during interactive pipeline runs when the next block
is ``ManualChannelExclude``.  Provides two modes:

**view+exclude** (montage available from SNIRF or sidecar)
    Renders the probe in its real optode positions.  Clicking a channel
    toggles exclusion on/off.  Excluded channels turn gray with an X marker.
    The status badge below the graph lists all currently excluded channels.

**positioning** (no montage found)
    No real positions exist yet.  A 2-click pattern lets the user place each
    optode:  first click selects an optode from a dropdown, second click
    (on the graph background) places it.  ``[Save positions]`` writes the
    sidecar JSON so subsequent runs open in view+exclude mode.

The dialog produces ``params_override = {"channels": [...]}`` — a list of
simple channel-name tokens (without wavelength suffix).
:class:`~nirspy.blocks.manual_exclude.ManualChannelExcludeBlock` already
expands S1_D2 → S1_D2 760 + S1_D2 850 internally (D7).

Design decisions
----------------
D5  Head shape: 2D top-view Plotly pure (ellipse + nose + ears).
D6  Positioning: persisted in ``<snirf>.montage.json`` sidecar.
D7  Dialog = probe-only + click-to-exclude, compact list below probe.
"""

from __future__ import annotations

import math
import re
from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from nirspy.io.montage import MontageDict, resolve_montage

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Badge colors for montage source labels
_SOURCE_BADGE_COLOR: dict[str, str] = {
    "snirf": "primary",
    "sidecar": "success",
    "missing": "warning",
}

_SOURCE_BADGE_LABEL: dict[str, str] = {
    "snirf": "Loaded from SNIRF",
    "sidecar": "Loaded from sidecar",
    "missing": "Manual positioning",
}

# Regex to extract "S<n>_D<m>" prefix from a full channel name like "S1_D2 760"
_CHANNEL_PREFIX_RE = re.compile(r"^(S\d+_D\d+)", re.IGNORECASE)


def _channel_prefix(ch_name: str) -> str:
    """Return 'S1_D2' from 'S1_D2 760', or the full name if no match."""
    m = _CHANNEL_PREFIX_RE.match(ch_name)
    return m.group(1) if m else ch_name


def _draw_head_silhouette_fig(fig: go.Figure, scale: float) -> None:
    """Add 2D top-view head silhouette traces to *fig*.

    Re-uses the same geometry as probe_viewer._draw_head_silhouette (T-026).
    Defined locally to avoid cross-component dependency.
    """
    if scale <= 0:
        scale = 0.05

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

    fig.add_annotation(
        x=0,
        y=-scale * 1.3,
        text="Iz",
        showarrow=False,
        font={"size": 10, "color": "#999999"},
    )


def _build_probe_figure(
    montage: MontageDict,
    excluded_prefixes: set[str],
) -> go.Figure:
    """Build a Plotly Figure for the probe viewer (view+exclude mode).

    Sources and detectors are rendered at their montage positions.  Channels
    whose prefix (``S<n>_D<m>``) is in *excluded_prefixes* are drawn gray
    with an X symbol.

    Parameters
    ----------
    montage:
        Dict with ``"sources"`` and ``"detectors"`` lists of ``[x, y]``.
    excluded_prefixes:
        Set of channel prefixes that are currently marked for exclusion.

    Returns
    -------
    go.Figure
    """
    sources: list[list[float]] = montage.get("sources", [])
    detectors: list[list[float]] = montage.get("detectors", [])

    all_x = [p[0] for p in sources] + [p[0] for p in detectors]
    all_y = [p[1] for p in sources] + [p[1] for p in detectors]
    if all_x:
        x_range = max(all_x) - min(all_x) if len(all_x) > 1 else 0.1
        y_range = max(all_y) - min(all_y) if len(all_y) > 1 else 0.1
        head_scale = max(x_range, y_range) * 0.6 or 0.05
    else:
        head_scale = 0.05

    fig = go.Figure()
    _draw_head_silhouette_fig(fig, head_scale)

    # Build channel traces — one point per source-detector pair would require
    # actual pairing info. Instead, we show sources + detectors as optodes;
    # channels are implicitly S<n>_D<m> pairs. The click target is the midpoint.
    # For simplicity we represent each unique S<n>_D<m> pair as a midpoint
    # marker and let the user click it to toggle exclusion.

    pairs = _derive_channel_pairs(sources, detectors)

    normal_x, normal_y, normal_labels = [], [], []
    excluded_x, excluded_y, excluded_labels = [], [], []

    for label, (x, y) in pairs.items():
        if label in excluded_prefixes:
            excluded_x.append(x)
            excluded_y.append(y)
            excluded_labels.append(label)
        else:
            normal_x.append(x)
            normal_y.append(y)
            normal_labels.append(label)

    # Source positions
    src_x = [float(p[0]) for p in sources]
    src_y = [float(p[1]) for p in sources]
    src_labels = [f"S{i + 1}" for i in range(len(sources))]
    if src_x:
        fig.add_trace(
            go.Scatter(
                x=src_x,
                y=src_y,
                mode="markers",
                name="Sources",
                marker={"color": "#d62728", "size": 10, "symbol": "circle"},
                text=src_labels,
                hoverinfo="text",
                customdata=src_labels,
            )
        )

    # Detector positions
    det_x = [float(p[0]) for p in detectors]
    det_y = [float(p[1]) for p in detectors]
    det_labels = [f"D{i + 1}" for i in range(len(detectors))]
    if det_x:
        fig.add_trace(
            go.Scatter(
                x=det_x,
                y=det_y,
                mode="markers",
                name="Detectors",
                marker={"color": "#1f77b4", "size": 10, "symbol": "square"},
                text=det_labels,
                hoverinfo="text",
                customdata=det_labels,
            )
        )

    # Active channels (midpoints)
    if normal_x:
        fig.add_trace(
            go.Scatter(
                x=normal_x,
                y=normal_y,
                mode="markers+text",
                name="Channels",
                marker={"color": "#2ca02c", "size": 8, "symbol": "circle"},
                text=normal_labels,
                textposition="top center",
                hoverinfo="text",
                customdata=normal_labels,
            )
        )

    # Excluded channels (midpoints — gray X)
    if excluded_x:
        fig.add_trace(
            go.Scatter(
                x=excluded_x,
                y=excluded_y,
                mode="markers+text",
                name="Excluded",
                marker={"color": "#999999", "size": 8, "symbol": "x"},
                text=excluded_labels,
                textposition="top center",
                hoverinfo="text",
                customdata=excluded_labels,
            )
        )

    fig.update_layout(
        title="Probe Layout — click channel to toggle exclusion",
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        height=450,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
        clickmode="event",
    )
    return fig


def _build_positioning_figure(
    montage: MontageDict,
    selected_optode: str | None = None,
) -> go.Figure:
    """Build a Plotly Figure for the positioning mode.

    Shows optodes that have been placed (with their current positions) and
    highlights the currently selected optode (awaiting second click).

    Parameters
    ----------
    montage:
        Partial montage dict — may have fewer sources/detectors than final.
    selected_optode:
        Label of the optode currently awaiting placement (e.g. "S1", "D3"),
        or *None*.

    Returns
    -------
    go.Figure
    """
    sources: list[list[float]] = montage.get("sources", [])
    detectors: list[list[float]] = montage.get("detectors", [])

    fig = go.Figure()
    _draw_head_silhouette_fig(fig, 0.1)

    src_x = [float(p[0]) for p in sources]
    src_y = [float(p[1]) for p in sources]
    src_labels = [f"S{i + 1}" for i in range(len(sources))]

    if src_x:
        fig.add_trace(
            go.Scatter(
                x=src_x,
                y=src_y,
                mode="markers+text",
                name="Sources (placed)",
                marker={"color": "#d62728", "size": 12, "symbol": "circle"},
                text=src_labels,
                textposition="top center",
                hoverinfo="text",
                customdata=src_labels,
            )
        )

    det_x = [float(p[0]) for p in detectors]
    det_y = [float(p[1]) for p in detectors]
    det_labels = [f"D{i + 1}" for i in range(len(detectors))]

    if det_x:
        fig.add_trace(
            go.Scatter(
                x=det_x,
                y=det_y,
                mode="markers+text",
                name="Detectors (placed)",
                marker={"color": "#1f77b4", "size": 12, "symbol": "square"},
                text=det_labels,
                textposition="top center",
                hoverinfo="text",
                customdata=det_labels,
            )
        )

    title = "Probe Positioning — select optode below, then click to place"
    if selected_optode:
        title = f"Probe Positioning — click to place {selected_optode}"

    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        height=450,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
        clickmode="event",
    )
    return fig


def _derive_channel_pairs(
    sources: list[list[float]],
    detectors: list[list[float]],
) -> dict[str, tuple[float, float]]:
    """Derive S<n>_D<m> channel pair midpoints from source/detector arrays.

    Each source is paired with each detector (fully connected), which is the
    common fNIRS arrangement. The midpoint of each pair is used as the
    click target in view+exclude mode.

    Returns
    -------
    dict mapping "S<n>_D<m>" -> (x_mid, y_mid)
    """
    pairs: dict[str, tuple[float, float]] = {}
    for i, src in enumerate(sources):
        for j, det in enumerate(detectors):
            label = f"S{i + 1}_D{j + 1}"
            x_mid = (float(src[0]) + float(det[0])) / 2.0
            y_mid = (float(src[1]) + float(det[1])) / 2.0
            pairs[label] = (x_mid, y_mid)
    return pairs


def _build_status_badge(excluded_prefixes: set[str]) -> html.Div:
    """Build a status badge listing currently excluded channels."""
    if not excluded_prefixes:
        return html.Div(
            dbc.Badge(
                "No channels excluded",
                color="success",
                className="me-1",
            ),
            id="probe-status-badge",
            className="mt-2",
        )

    badges = [
        dbc.Badge(prefix, color="secondary", className="me-1")
        for prefix in sorted(excluded_prefixes)
    ]
    return html.Div(
        [html.Small("Excluded: ", className="text-muted")] + badges,
        id="probe-status-badge",
        className="mt-2",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_probe_dialog(
    snirf_path: str | None,
    current_idx: int,
    total: int,
    excluded_channels: list[str] | None = None,
    positioned_montage: MontageDict | None = None,
    selected_optode: str | None = None,
) -> dbc.Modal:
    """Build the probe interactive dialog Modal.

    Resolves the montage via ``resolve_montage()`` (sidecar > SNIRF > None).
    If a montage is available, renders in **view+exclude** mode.
    Otherwise renders in **positioning** mode.

    Parameters
    ----------
    snirf_path:
        Path to the SNIRF file loaded in the current run, or *None*.
    current_idx:
        0-based index of the current block among enabled steps.
    total:
        Total number of enabled steps in the pipeline.
    excluded_channels:
        List of channel prefixes (e.g. ``["S1_D2"]``) currently excluded.
        These are channel name prefixes — wavelength expansion is handled
        by :class:`~nirspy.blocks.manual_exclude.ManualChannelExcludeBlock`.
    positioned_montage:
        In-progress montage dict for positioning mode (may be partially filled).
        When *None* the dialog starts with an empty positioning canvas.
    selected_optode:
        In positioning mode, the optode currently awaiting its second click.

    Returns
    -------
    dbc.Modal
        Fully configured modal, initially open (``is_open=True``).
    """
    excluded_set: set[str] = set(excluded_channels or [])

    # Resolve montage
    montage: MontageDict | None = None
    source_label = "missing"
    if snirf_path:
        montage, source_label = resolve_montage(snirf_path)

    # Use in-progress positioned montage if passed (positioning mode state)
    if montage is None and positioned_montage is not None:
        montage = positioned_montage

    view_mode = montage is not None  # True = view+exclude; False = positioning

    # --- Header ---
    step_num = current_idx + 1
    snirf_name = snirf_path.split("\\")[-1].split("/")[-1] if snirf_path else "SNIRF"
    badge_color = _SOURCE_BADGE_COLOR.get(source_label, "secondary")
    badge_text = _SOURCE_BADGE_LABEL.get(source_label, source_label)

    header_content = dbc.Row(
        [
            dbc.Col(
                dbc.ModalTitle(
                    f"Block {step_num}/{total}: Manual Channel Exclude"
                ),
                width="auto",
            ),
            dbc.Col(
                [
                    html.Small(snirf_name, className="text-muted me-2"),
                    dbc.Badge(badge_text, color=badge_color, pill=True),
                ],
                className="d-flex align-items-center",
            ),
        ],
        align="center",
        className="w-100",
    )

    # --- Body: probe graph ---
    if view_mode:
        assert montage is not None  # guaranteed by view_mode logic above
        fig = _build_probe_figure(montage, excluded_set)
        mode_hint = html.P(
            "Click a channel midpoint (green circle) to toggle exclusion.",
            className="text-muted small mb-1",
        )
    else:
        positioning_montage: MontageDict = positioned_montage or {"sources": [], "detectors": []}
        fig = _build_positioning_figure(positioning_montage, selected_optode)
        mode_hint = html.P(
            "No montage found. Select an optode below and click on the plot to place it.",
            className="text-warning small mb-1",
        )

    probe_graph = dcc.Graph(
        id="probe-dialog-graph",
        figure=fig,
        config={"displayModeBar": False},
        style={"height": "460px"},
    )

    # Hidden stores for client-side state
    excluded_store = dcc.Store(
        id="probe-excluded-store",
        data=list(excluded_set),
    )
    snirf_store = dcc.Store(
        id="probe-snirf-path-store",
        data=snirf_path,
    )
    mode_store = dcc.Store(
        id="probe-mode-store",
        data="view" if view_mode else "positioning",
    )
    selected_optode_store = dcc.Store(
        id="probe-selected-optode-store",
        data=selected_optode,
    )
    positioned_store = dcc.Store(
        id="probe-positioned-montage-store",
        data=positioned_montage or {"sources": [], "detectors": []},
    )

    status_badge = _build_status_badge(excluded_set)

    # Positioning-mode extras: optode selector + save button
    positioning_controls: list[Any] = []
    if not view_mode:
        n_sources = len((positioned_montage or {}).get("sources", []))
        n_detectors = len((positioned_montage or {}).get("detectors", []))
        # Provide a simple list of optode labels for selection
        optode_options = [
            {"label": f"S{i + 1}", "value": f"S{i + 1}"}
            for i in range(max(n_sources + 1, 4))
        ] + [
            {"label": f"D{i + 1}", "value": f"D{i + 1}"}
            for i in range(max(n_detectors + 1, 4))
        ]
        positioning_controls = [
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Select(
                            id="probe-optode-selector",
                            options=optode_options,
                            placeholder="Select optode to place...",
                            value=selected_optode,
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Save positions",
                            id="probe-save-positions-btn",
                            color="success",
                            outline=True,
                            size="sm",
                        ),
                        width="auto",
                    ),
                ],
                className="mt-2",
            ),
        ]

    body_children: list[Any] = [
        excluded_store,
        snirf_store,
        mode_store,
        selected_optode_store,
        positioned_store,
        mode_hint,
        probe_graph,
        status_badge,
        *positioning_controls,
    ]

    # --- Footer ---
    footer = dbc.ModalFooter(
        dbc.ButtonGroup(
            [
                dbc.Button(
                    "Cancel run",
                    id="probe-cancel-btn",
                    color="danger",
                    outline=True,
                    size="sm",
                ),
                dbc.Button(
                    "Confirm exclusions & continue",
                    id="probe-confirm-btn",
                    color="primary",
                    size="sm",
                ),
            ],
            size="sm",
        )
    )

    return dbc.Modal(
        [
            dbc.ModalHeader(header_content, close_button=False),
            dbc.ModalBody(body_children),
            footer,
        ],
        id="probe-dialog-modal",
        is_open=True,
        backdrop="static",
        keyboard=False,
        size="xl",
    )


def build_channels_override(excluded_prefixes: list[str]) -> dict[str, list[str]]:
    """Build ``params_override`` from a list of excluded channel prefixes.

    The ManualChannelExcludeBlock accepts simple channel name prefixes and
    expands them to wavelength pairs internally.

    Parameters
    ----------
    excluded_prefixes:
        List such as ``["S1_D2", "S3_D1"]``.

    Returns
    -------
    dict with key ``"channels"`` containing the list of prefixes.
    """
    return {"channels": list(excluded_prefixes)}
