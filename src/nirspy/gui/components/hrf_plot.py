"""HRF plot component — condition-averaged HbO/HbR with error bands."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from dash import dcc, html

# Conventional colours (Yücel et al., 2021 / project convention)
_HBO_COLOR = "#d62728"
_HBR_COLOR = "#1f77b4"


def render_hrf_plot(
    evoked_dict: dict[str, Any] | None = None,
    selected_conditions: list[str] | None = None,
) -> html.Div:
    """Render HRF waveforms (HbO red, HbR blue) per condition.

    Parameters
    ----------
    evoked_dict:
        Mapping ``{condition_name: mne.Evoked}``.
    selected_conditions:
        Subset of condition names to display.

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

        # Separate HbO and HbR channels
        hbo_idx = [
            i for i, ch in enumerate(ch_names)
            if "hbo" in ch.lower()
        ]
        hbr_idx = [
            i for i, ch in enumerate(ch_names)
            if "hbr" in ch.lower()
        ]

        for idx_list, color, label in [
            (hbo_idx, _HBO_COLOR, "HbO"),
            (hbr_idx, _HBR_COLOR, "HbR"),
        ]:
            if not idx_list:
                continue

            mean_signal = np.mean(data[idx_list], axis=0)
            name = f"{cond} — {label}"

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
                )
                upper = (mean_signal + sem).tolist()
                lower = (mean_signal - sem).tolist()

                fig.add_trace(
                    go.Scatter(
                        x=times.tolist() + times[::-1].tolist(),
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor=_rgba(color, 0.15),
                        line={"color": "rgba(0,0,0,0)"},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        title="HRF — Haemodynamic Response",
        xaxis_title="Time (s)",
        yaxis_title="Concentration",
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
