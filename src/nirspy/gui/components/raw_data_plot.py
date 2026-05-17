"""Raw data time-series plot component."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from dash import dcc, html

# Maximum channels rendered by default to keep the plot responsive.
_MAX_DEFAULT_CHANNELS = 10


def render_raw_data_plot(
    raw_data: Any = None,
    selected_channels: list[str] | None = None,
) -> html.Div:
    """Render a multi-channel time-series plot from an MNE Raw object.

    Parameters
    ----------
    raw_data:
        An ``mne.io.BaseRaw`` instance, or *None* for placeholder.
    selected_channels:
        Subset of channel names to display.  When *None*, the first
        ``_MAX_DEFAULT_CHANNELS`` channels are shown.

    Returns
    -------
    html.Div
        Plotly Graph wrapped in a container div.
    """
    if raw_data is None:
        return html.Div(
            html.P(
                "No raw data available. Run the pipeline first.",
                className="text-muted text-center py-4",
            ),
            id="raw-data-plot-content",
        )

    # Import MNE lazily to keep module import lightweight.
    import mne

    if not isinstance(raw_data, mne.io.BaseRaw):
        return html.Div(
            html.P(
                "Data is not a Raw object.",
                className="text-muted text-center py-4",
            ),
            id="raw-data-plot-content",
        )

    ch_names: list[str] = list(raw_data.ch_names)
    if selected_channels:
        plot_chs = [c for c in selected_channels if c in ch_names]
    else:
        plot_chs = ch_names[:_MAX_DEFAULT_CHANNELS]

    if not plot_chs:
        plot_chs = ch_names[:_MAX_DEFAULT_CHANNELS]

    times = raw_data.times
    picks = [ch_names.index(c) for c in plot_chs]
    data_array, _ = raw_data[picks, :]

    fig = go.Figure()
    for i, ch in enumerate(plot_chs):
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=data_array[i],
                mode="lines",
                name=ch,
                line={"width": 1},
            )
        )

    fig.update_layout(
        title="Raw Time Series",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=400,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.2},
        template="plotly_white",
    )

    return html.Div(
        dcc.Graph(
            id="raw-data-graph",
            figure=fig,
            config={"scrollZoom": True},
        ),
        id="raw-data-plot-content",
    )
