"""QC dashboard component — SCI heatmap per channel."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from dash import dcc, html

_SCI_THRESHOLD = 0.7


def render_qc_dashboard(
    sci_values: Any = None,
    ch_names: list[str] | None = None,
) -> html.Div:
    """Render a heatmap of Scalp Coupling Index values.

    Channels with SCI >= 0.7 are green; below 0.7 are red.

    Parameters
    ----------
    sci_values:
        1-D array-like of SCI values, one per channel.
    ch_names:
        List of channel names matching *sci_values*.

    Returns
    -------
    html.Div
        Plotly heatmap or placeholder.
    """
    if sci_values is None:
        return html.Div(
            html.P(
                "No QC data available. Run a pipeline with "
                "Scalp Coupling Index block.",
                className="text-muted text-center py-4",
            ),
            id="qc-dashboard-content",
        )

    import numpy as np

    values = np.asarray(sci_values, dtype=float)
    if values.ndim == 0 or values.size == 0:
        return html.Div(
            html.P(
                "No QC data available.",
                className="text-muted text-center py-4",
            ),
            id="qc-dashboard-content",
        )

    labels = ch_names or [f"Ch {i}" for i in range(len(values))]

    # Reshape to 2D for heatmap (single row)
    z = values.reshape(1, -1)

    colorscale = [
        [0.0, "#dc3545"],
        [_SCI_THRESHOLD - 0.001, "#dc3545"],
        [_SCI_THRESHOLD, "#28a745"],
        [1.0, "#28a745"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z.tolist(),
            x=labels,
            y=["SCI"],
            colorscale=colorscale,
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "SCI"},
            hovertemplate=(
                "Channel: %{x}<br>"
                "SCI: %{z:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Scalp Coupling Index (threshold {_SCI_THRESHOLD})",
        height=200,
        margin={"l": 60, "r": 20, "t": 40, "b": 60},
        xaxis={"tickangle": -45},
        template="plotly_white",
    )

    return html.Div(
        dcc.Graph(
            id="qc-dashboard-graph",
            figure=fig,
        ),
        id="qc-dashboard-content",
    )
