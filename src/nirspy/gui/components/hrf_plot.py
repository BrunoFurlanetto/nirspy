"""HRF plot component -- condition-averaged HbO/HbR with error bands.

Values are scaled from mol/L to micromolar (uM) for display, following
standard fNIRS reporting conventions.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from dash import dcc, html

# Conventional colours (Yuecel et al., 2021 / project convention)
_HBO_COLOR = "#d62728"
_HBR_COLOR = "#1f77b4"

# Scaling factor: mol/L -> micromolar (uM)
_MOL_TO_MICROMOLAR: float = 1e6

# Discard-region overlay defaults (seconds)
_DISCARD_TMIN_DEFAULT: float = -5.0
_DISCARD_TMAX_DEFAULT: float = 5.0


def render_hrf_plot(
    evoked_dict: dict[str, Any] | None = None,
    selected_conditions: list[str] | None = None,
    discard_toggle: bool = False,
    discard_tmin: float | None = None,
    discard_tmax: float | None = None,
) -> html.Div:
    """Render HRF waveforms (HbO red, HbR blue) per condition.

    Values are scaled to micromolar (uM) for display.  Legend entries
    use ``Delta uM`` notation.

    An optional *discard region* overlay can be shown as a semi-transparent
    red rectangle between ``discard_tmin`` and ``discard_tmax``.  This is
    **purely visual** — it does not affect baseline correction or the
    BlockAverage computation in any way.  It is intended to help the user
    identify time windows that should be excluded when interpreting the
    waveform (e.g. a pre-stimulus baseline artifact window).

    Parameters
    ----------
    evoked_dict:
        Mapping ``{condition_name: mne.Evoked}``.
    selected_conditions:
        Subset of condition names to display.
    discard_toggle:
        Whether to show the discard region overlay.
    discard_tmin:
        Left boundary of the discard region (seconds).
        Defaults to ``-5`` when ``None``.
    discard_tmax:
        Right boundary of the discard region (seconds).
        Defaults to ``5`` when ``None``.

    Returns
    -------
    html.Div
        Plotly Graph or placeholder.
    """
    if not evoked_dict:
        return html.Div(
            html.P(
                "No HRF data available. Run a pipeline with "
                "Block Average.",
                className="text-muted text-center py-4",
            ),
            id="hrf-plot-content",
        )

    import numpy as np

    conditions = (
        selected_conditions
        if selected_conditions
        else list(evoked_dict.keys())
    )

    fig = go.Figure()

    for cond in conditions:
        evoked = evoked_dict.get(cond)
        if evoked is None:
            continue

        times = evoked.times
        data = evoked.data  # (n_channels, n_times)
        ch_names = evoked.ch_names
        nave = getattr(evoked, "nave", 1)
        bads = set(evoked.info.get("bads", []))

        # Separate HbO and HbR channels, excluding bads
        hbo_idx = [
            i for i, ch in enumerate(ch_names)
            if "hbo" in ch.lower() and ch not in bads
        ]
        hbr_idx = [
            i for i, ch in enumerate(ch_names)
            if "hbr" in ch.lower() and ch not in bads
        ]

        for idx_list, color, label in [
            (hbo_idx, _HBO_COLOR, "HbO"),
            (hbr_idx, _HBR_COLOR, "HbR"),
        ]:
            if not idx_list:
                continue

            mean_signal = (
                np.mean(data[idx_list], axis=0)
                * _MOL_TO_MICROMOLAR
            )
            name = f"{cond} — {label} ΔμM"

            fig.add_trace(
                go.Scatter(
                    x=times.tolist(),
                    y=mean_signal.tolist(),
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 2},
                )
            )

            # SEM error band if nave > 1
            if nave > 1 and len(idx_list) > 1:
                sem = (
                    np.std(data[idx_list], axis=0)
                    / np.sqrt(nave)
                    * _MOL_TO_MICROMOLAR
                )
                upper = (mean_signal + sem).tolist()
                lower = (mean_signal - sem).tolist()

                fig.add_trace(
                    go.Scatter(
                        x=times.tolist()
                        + times[::-1].tolist(),
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor=_rgba(color, 0.15),
                        line={"color": "rgba(0,0,0,0)"},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # Discard-region overlay — purely visual, no data impact
    if discard_toggle:
        t0 = discard_tmin if discard_tmin is not None else _DISCARD_TMIN_DEFAULT
        t1 = discard_tmax if discard_tmax is not None else _DISCARD_TMAX_DEFAULT
        if t0 < t1:
            fig.add_vrect(
                x0=t0,
                x1=t1,
                fillcolor="rgba(255,0,0,0.15)",
                line_width=0,
                layer="below",
                annotation_text="ignore range",
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color="rgba(200,0,0,0.7)",
            )

    fig.update_layout(
        title="HRF — Haemodynamic Response",
        xaxis_title="Time (s)",
        yaxis_title="Concentration (μM)",
        height=400,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.2},
        template="plotly_white",
    )

    return html.Div(
        dcc.Graph(
            id="hrf-graph",
            figure=fig,
        ),
        id="hrf-plot-content",
    )


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to ``rgba(R,G,B,a)``."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
