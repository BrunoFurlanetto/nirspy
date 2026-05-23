"""Probe interactive dialog for ManualChannelExclude (T-029).

Head silhouette uses a fixed canonical size (radius 1.1 x 1.2). Optode
positions read from montage are normalised (centered + scaled) to fit
inside the canonical head so the probe always lands in the correct
anatomical region regardless of the raw position units (mm/cm/m).

A faint 10-20 system overlay is rendered behind the optodes as a
reference grid. It can be toggled off via the dialog controls.

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


# Canonical head silhouette dimensions (unitless — viewport coordinates).
# rx slightly less than ry to match the human head ellipse top-view.
_HEAD_RX: float = 1.1
_HEAD_RY: float = 1.2

# Target radius for normalised optode placement (90% of head).
_OPTODE_FIT_RADIUS: float = 1.0

# 10-20 system reference positions (top-view, head-normalised).
# Approximated from standard EEG 10-20 layout. Used as a faint overlay
# so the user can verify probe optodes land near the intended region.
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
    """Return the angle (radians) of the principal axis vs. the x-axis.

    Uses the closed-form 2x2 eigen-decomposition of the sample
    covariance. The result is in ``[-pi/2, pi/2]``. Subtracting it from
    every point rotates the cloud so its longest axis aligns with x.
    """
    if len(centered_pts) < 2:
        return 0.0
    sxx = sum(p[0] * p[0] for p in centered_pts)
    syy = sum(p[1] * p[1] for p in centered_pts)
    sxy = sum(p[0] * p[1] for p in centered_pts)
    # Angle of the dominant eigenvector for 2x2 symmetric matrix
    # [[sxx, sxy],[sxy, syy]] is 0.5 * atan2(2*sxy, sxx - syy).
    return 0.5 * math.atan2(2.0 * sxy, sxx - syy)


def _normalize_positions(
    sources: list[list[float]],
    detectors: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """Recenter, auto-rotate and rescale optode positions.

    Steps:
      1. Subtract combined centroid (probe centered at origin).
      2. PCA rotation so the probe's long axis aligns with x (the
         horizontal axis of the head silhouette). Asymmetric probes
         that were stored vertically in the SNIRF file thus land
         horizontally on the head, matching the typical mounting.
      3. Uniform scale so the farthest optode sits at
         ``_OPTODE_FIT_RADIUS``.

    Returns
    -------
    (sources_norm, detectors_norm)
        Same shape as inputs, in head-normalised coordinates.
    """
    if not sources and not detectors:
        return [], []

    all_pts = [(float(p[0]), float(p[1])) for p in sources + detectors]
    cx = sum(p[0] for p in all_pts) / len(all_pts)
    cy = sum(p[1] for p in all_pts) / len(all_pts)

    centered = [(p[0] - cx, p[1] - cy) for p in all_pts]

    theta = _pca_rotation_angle(centered)
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)

    rotated = [
        (p[0] * cos_t - p[1] * sin_t, p[0] * sin_t + p[1] * cos_t)
        for p in centered
    ]

    max_r = max(math.hypot(p[0], p[1]) for p in rotated)
    scale = 1.0 if max_r <= 0 else _OPTODE_FIT_RADIUS / max_r

    def _apply(pts: list[list[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for p in pts:
            x = float(p[0]) - cx
            y = float(p[1]) - cy
            xr = x * cos_t - y * sin_t
            yr = x * sin_t + y * cos_t
            out.append([xr * scale, yr * scale])
        return out

    return _apply(sources), _apply(detectors)


_OPTODE_SHAPE_RADIUS: float = 0.06

#: Customdata prefix used on the 10-20 scatter trace so the click
#: callback can distinguish a reference click from a channel-midpoint
#: click.
_TEN_TWENTY_CD_PREFIX: str = "1020:"


def normalize_montage(montage: MontageDict) -> MontageDict:
    """Return a montage with PCA-rotated + scaled positions for plotting."""
    s, d = _normalize_positions(
        montage.get("sources", []),
        montage.get("detectors", []),
    )
    return {"sources": s, "detectors": d}


def _add_optode_shapes(
    fig: go.Figure,
    sources: list[list[float]],
    detectors: list[list[float]],
) -> None:
    """Add editable circle (source) / square (detector) shapes + labels.

    Shape ordering is stable: indices ``0..len(sources)-1`` are sources,
    the rest are detectors. ``relayoutData`` events use these indices.
    """
    r = _OPTODE_SHAPE_RADIUS
    for i, (x, y) in enumerate(sources):
        fig.add_shape(
            type="circle",
            x0=x - r,
            y0=y - r,
            x1=x + r,
            y1=y + r,
            fillcolor="rgba(214,39,40,0.85)",
            line={"color": "#d62728", "width": 1},
            layer="above",
        )
        fig.add_annotation(
            x=x,
            y=y + r + 0.06,
            text=f"S{i + 1}",
            showarrow=False,
            font={"size": 10, "color": "#d62728"},
        )
    for i, (x, y) in enumerate(detectors):
        fig.add_shape(
            type="rect",
            x0=x - r,
            y0=y - r,
            x1=x + r,
            y1=y + r,
            fillcolor="rgba(31,119,180,0.85)",
            line={"color": "#1f77b4", "width": 1},
            layer="above",
        )
        fig.add_annotation(
            x=x,
            y=y + r + 0.06,
            text=f"D{i + 1}",
            showarrow=False,
            font={"size": 10, "color": "#1f77b4"},
        )


def translate_probe_to_anchor(
    montage: MontageDict,
    channel_label: str,
    target_xy: tuple[float, float],
) -> MontageDict:
    """Translate the whole probe so a channel midpoint lands at *target_xy*.

    Computes the current midpoint of ``channel_label`` (e.g. ``"S1_D2"``)
    and shifts every optode by the delta required to put that midpoint
    on top of *target_xy*. The probe geometry (relative S-D distances)
    is preserved.
    """
    if not channel_label or "_" not in channel_label:
        return montage
    s_part, d_part = channel_label.split("_", 1)
    try:
        src_idx = int(s_part[1:]) - 1
        det_idx = int(d_part[1:]) - 1
    except ValueError:
        return montage

    sources = [list(p) for p in montage.get("sources", [])]
    detectors = [list(p) for p in montage.get("detectors", [])]
    if not (0 <= src_idx < len(sources) and 0 <= det_idx < len(detectors)):
        return montage

    sx, sy = sources[src_idx]
    dx_, dy_ = detectors[det_idx]
    mx = (sx + dx_) / 2.0
    my = (sy + dy_) / 2.0
    dx = target_xy[0] - mx
    dy = target_xy[1] - my

    return {
        "sources": [[p[0] + dx, p[1] + dy] for p in sources],
        "detectors": [[p[0] + dx, p[1] + dy] for p in detectors],
    }




def _channel_midpoint(
    montage: MontageDict,
    channel_label: str,
) -> tuple[float, float] | None:
    """Return the midpoint (x, y) for a channel label like ``"S1_D2"``."""
    if not channel_label or "_" not in channel_label:
        return None
    s_part, d_part = channel_label.split("_", 1)
    try:
        src_idx = int(s_part[1:]) - 1
        det_idx = int(d_part[1:]) - 1
    except ValueError:
        return None
    sources = montage.get("sources", [])
    detectors = montage.get("detectors", [])
    if not (0 <= src_idx < len(sources) and 0 <= det_idx < len(detectors)):
        return None
    sx, sy = float(sources[src_idx][0]), float(sources[src_idx][1])
    dx_, dy_ = float(detectors[det_idx][0]), float(detectors[det_idx][1])
    return ((sx + dx_) / 2.0, (sy + dy_) / 2.0)


def _apply_similarity(
    pts: list[list[float]],
    scale: float,
    cos_t: float,
    sin_t: float,
    tx: float,
    ty: float,
) -> list[list[float]]:
    """Apply a 2-D similarity transform to a list of [x, y] points."""
    out: list[list[float]] = []
    for p in pts:
        x, y = float(p[0]), float(p[1])
        xr = scale * (cos_t * x - sin_t * y) + tx
        yr = scale * (sin_t * x + cos_t * y) + ty
        out.append([xr, yr])
    return out


def similarity_transform_probe(
    montage: MontageDict,
    anchors: list[list[str]],
) -> MontageDict:
    """Apply a similarity transform so channel midpoints best-fit 10-20 targets.

    - **0 anchors:** unchanged.
    - **1 anchor:** pure translation.
    - **2 anchors:** exact similarity (4 DOF).
    - **3+ anchors:** Umeyama / Horn least-squares fit.
    """
    if not anchors:
        return montage

    src_pts: list[tuple[float, float]] = []
    tgt_pts: list[tuple[float, float]] = []
    for pair in anchors:
        ch_label, ref_label = pair[0], pair[1]
        mid = _channel_midpoint(montage, ch_label)
        if mid is None:
            continue
        target = _TEN_TWENTY.get(ref_label)
        if target is None:
            continue
        src_pts.append(mid)
        tgt_pts.append(target)

    if not src_pts:
        return montage

    n = len(src_pts)

    if n == 1:
        return translate_probe_to_anchor(montage, anchors[0][0], tgt_pts[0])

    # Umeyama-style similarity transform (works for n >= 2)
    cx_s = sum(p[0] for p in src_pts) / n
    cy_s = sum(p[1] for p in src_pts) / n
    cx_t = sum(p[0] for p in tgt_pts) / n
    cy_t = sum(p[1] for p in tgt_pts) / n

    qs = [(p[0] - cx_s, p[1] - cy_s) for p in src_pts]
    qt = [(p[0] - cx_t, p[1] - cy_t) for p in tgt_pts]

    sxx = sum(q[0] * t[0] for q, t in zip(qs, qt, strict=True))
    sxy = sum(q[0] * t[1] for q, t in zip(qs, qt, strict=True))
    syx = sum(q[1] * t[0] for q, t in zip(qs, qt, strict=True))
    syy = sum(q[1] * t[1] for q, t in zip(qs, qt, strict=True))

    theta = math.atan2(sxy - syx, sxx + syy)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    var_s = sum(q[0] ** 2 + q[1] ** 2 for q in qs)
    if var_s < 1e-15:
        return translate_probe_to_anchor(montage, anchors[0][0], tgt_pts[0])

    rot_dot = sum(
        (cos_t * q[0] - sin_t * q[1]) * t[0]
        + (sin_t * q[0] + cos_t * q[1]) * t[1]
        for q, t in zip(qs, qt, strict=True)
    )
    scale = rot_dot / var_s

    sources = [list(p) for p in montage.get("sources", [])]
    detectors = [list(p) for p in montage.get("detectors", [])]

    tx = cx_t - scale * (cos_t * cx_s - sin_t * cy_s)
    ty = cy_t - scale * (sin_t * cx_s + cos_t * cy_s)

    return {
        "sources": _apply_similarity(sources, scale, cos_t, sin_t, tx, ty),
        "detectors": _apply_similarity(
            detectors, scale, cos_t, sin_t, tx, ty
        ),
    }


def _draw_anchor_lines(
    fig: go.Figure,
    montage: MontageDict,
    anchors: list[list[str]],
) -> None:
    """Draw dashed lines from each anchored channel midpoint to target."""
    for pair in anchors:
        ch_label, ref_label = pair[0], pair[1]
        mid = _channel_midpoint(montage, ch_label)
        target = _TEN_TWENTY.get(ref_label)
        if mid is None or target is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=[mid[0], target[0]],
                y=[mid[1], target[1]],
                mode="lines",
                line={
                    "color": "rgba(255,140,0,0.6)",
                    "width": 1.5,
                    "dash": "dash",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _build_anchor_badges(
    anchors: list[list[str]],
) -> html.Div:
    """Build a row of small badges showing current anchors."""
    if not anchors:
        return html.Div(
            dbc.Badge(
                "No anchors set",
                color="info",
                className="me-1",
            ),
            id="probe-anchor-badges",
            className="mt-1 mb-1",
        )
    badges = [
        dbc.Badge(
            f"{pair[0]} > {pair[1]}",
            color="warning",
            text_color="dark",
            className="me-1",
        )
        for pair in anchors
    ]
    return html.Div(
        [html.Small("Anchors: ", className="text-muted")] + badges,
        id="probe-anchor-badges",
        className="mt-1 mb-1",
    )


def _draw_ten_twenty_overlay(fig: go.Figure) -> None:
    """Add a clickable 10-20 reference layer behind the optodes.

    Each marker carries customdata ``"1020:<label>"`` so the click
    callback can distinguish a 10-20 target click from a channel
    midpoint click.
    """
    xs = [v[0] for v in _TEN_TWENTY.values()]
    ys = [v[1] for v in _TEN_TWENTY.values()]
    labels = list(_TEN_TWENTY.keys())
    cd = [f"{_TEN_TWENTY_CD_PREFIX}{lbl}" for lbl in labels]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            name="10-20 reference",
            marker={
                "color": "rgba(120,120,120,0.45)",
                "size": 10,
                "symbol": "circle-open",
                "line": {"width": 1.5},
            },
            text=labels,
            textfont={"color": "rgba(80,80,80,0.7)", "size": 9},
            textposition="bottom center",
            hoverinfo="text",
            customdata=cd,
            showlegend=True,
            visible=True,
        )
    )


def _channel_prefix(ch_name: str) -> str:
    """Return 'S1_D2' from 'S1_D2 760', or the full name if no match."""
    m = _CHANNEL_PREFIX_RE.match(ch_name)
    return m.group(1) if m else ch_name


def _draw_head_silhouette_fig(fig: go.Figure, scale: float = 1.0) -> None:
    """Add 2D top-view head silhouette traces to *fig*.

    Renders at the canonical head size (``_HEAD_RX`` x ``_HEAD_RY``).
    The ``scale`` argument is kept for backward compatibility but
    ignored — positions are normalised before plotting (see
    :func:`_normalize_positions`).
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


def _build_probe_figure(
    montage: MontageDict,
    excluded_prefixes: set[str],
    show_ten_twenty: bool = True,
    selected_channel: str | None = None,
    anchors: list[list[str]] | None = None,
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
    raw_sources: list[list[float]] = montage.get("sources", [])
    raw_detectors: list[list[float]] = montage.get("detectors", [])

    # If the caller has already passed a normalised montage (after drag
    # updates), the second normalize is idempotent up to PCA flips. To
    # avoid jitter we skip when positions are already inside the
    # canonical head (max radius < 1.1).
    all_pts = raw_sources + raw_detectors
    if all_pts and max(
        math.hypot(p[0], p[1]) for p in all_pts
    ) <= _HEAD_RX + 0.01:
        sources = [list(p) for p in raw_sources]
        detectors = [list(p) for p in raw_detectors]
    else:
        sources, detectors = _normalize_positions(raw_sources, raw_detectors)

    fig = go.Figure()
    _draw_head_silhouette_fig(fig)
    if show_ten_twenty:
        _draw_ten_twenty_overlay(fig)

    # Draw dashed anchor lines from channel midpoints to 10-20 targets
    if anchors:
        _draw_anchor_lines(fig, montage, anchors)

    # Channel midpoints (full all-pairs S x D since MeasList is not
    # carried in MontageDict; the click target lets the user toggle
    # exclusion).
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

    # Optode shapes (editable - dragged by the user to refine positions).
    _add_optode_shapes(fig, sources, detectors)

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

    # Highlight the currently-selected channel so the user knows which
    # midpoint will move when they click a 10-20 target next.
    if selected_channel and selected_channel in pairs:
        sx, sy = pairs[selected_channel]
        fig.add_trace(
            go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers+text",
                name="Selected",
                marker={
                    "color": "rgba(255,165,0,0.0)",
                    "size": 22,
                    "symbol": "circle",
                    "line": {"color": "#ff8c00", "width": 3},
                },
                text=[selected_channel],
                textposition="bottom center",
                textfont={"color": "#ff8c00", "size": 11},
                hoverinfo="text",
                customdata=[selected_channel],
            )
        )

    title_msg = (
        f"Click a 10-20 target to place {selected_channel} midpoint"
        if selected_channel
        else "Click a channel (green) to position, then a 10-20 to place"
    )
    fig.update_layout(
        title=title_msg,
        xaxis={
            "visible": False,
            "range": [-(_HEAD_RX + 0.25), _HEAD_RX + 0.25],
        },
        yaxis={
            "visible": False,
            "scaleanchor": "x",
            "range": [-(_HEAD_RY + 0.3), _HEAD_RY + 0.3],
        },
        height=450,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
        clickmode="event",
    )
    return fig


def _build_positioning_figure(
    montage: MontageDict,
    selected_optode: str | None = None,
    show_ten_twenty: bool = True,
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
    _draw_head_silhouette_fig(fig)
    if show_ten_twenty:
        _draw_ten_twenty_overlay(fig)

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
        xaxis={
            "visible": False,
            "range": [-(_HEAD_RX + 0.25), _HEAD_RX + 0.25],
        },
        yaxis={
            "visible": False,
            "scaleanchor": "x",
            "range": [-(_HEAD_RY + 0.3), _HEAD_RY + 0.3],
        },
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
        # Normalise once up-front so the store and figure share frame.
        normalised = normalize_montage(montage)
        montage = normalised
        fig = _build_probe_figure(montage, excluded_set)
        mode_hint = html.P(
            [
                "Multi-anchor positioning: ",
                html.B("(1) "),
                "click a green channel midpoint; ",
                html.B("(2) "),
                "click a 10-20 target to anchor it. "
                "Repeat to add more anchors. "
                "1 anchor = translate, 2 = rotate+scale, 3+ = best fit. "
                "Use the checklist below to exclude noisy channels.",
            ],
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
    # Seed the positioned-montage store with whatever the figure is
    # rendering: in view+exclude mode this is the normalised SNIRF
    # montage so drag callbacks can edit the same coordinates the user
    # sees on screen.
    if view_mode and montage is not None:
        store_montage = montage
    else:
        store_montage = positioned_montage or {"sources": [], "detectors": []}
    positioned_store = dcc.Store(
        id="probe-positioned-montage-store",
        data=store_montage,
    )
    # Selected channel for the 2-click positioning workflow.
    selected_channel_store = dcc.Store(
        id="probe-selected-channel-store",
        data=None,
    )
    # Anchor store: list of [channel_label, ten_twenty_label] pairs.
    anchors_store = dcc.Store(
        id="probe-anchors-store",
        data=[],
    )

    anchor_badges = _build_anchor_badges([])

    status_badge = _build_status_badge(excluded_set)

    # Exclusion checklist (view+exclude mode only).
    exclusion_controls: list[Any] = []
    if view_mode and montage is not None:
        sources_n = len(montage.get("sources", []))
        detectors_n = len(montage.get("detectors", []))
        channel_labels = [
            f"S{i + 1}_D{j + 1}"
            for i in range(sources_n)
            for j in range(detectors_n)
        ]
        exclusion_controls = [
            html.Hr(),
            html.Label(
                "Exclude channels:",
                className="fw-bold small mb-1",
            ),
            dbc.Checklist(
                id="probe-exclude-checklist",
                options=[{"label": lbl, "value": lbl} for lbl in channel_labels],
                value=sorted(excluded_set),
                inline=True,
                switch=False,
                className="small",
            ),
        ]

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

    # Reset-anchors button (view+exclude mode only)
    anchor_controls: list[Any] = []
    if view_mode:
        anchor_controls = [
            html.Div(
                [
                    anchor_badges,
                    dbc.Button(
                        "Reset anchors",
                        id="probe-reset-anchors-btn",
                        color="secondary",
                        outline=True,
                        size="sm",
                        className="ms-2",
                    ),
                ],
                className="d-flex align-items-center mt-1 mb-2",
            ),
        ]

    body_children: list[Any] = [
        excluded_store,
        snirf_store,
        mode_store,
        selected_optode_store,
        selected_channel_store,
        positioned_store,
        anchors_store,
        mode_hint,
        probe_graph,
        *anchor_controls,
        status_badge,
        *exclusion_controls,
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
