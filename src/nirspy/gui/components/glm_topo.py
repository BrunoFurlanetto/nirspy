"""GLM visualization component (T-036).

Provides two public functions:

- ``render_glm_topo(glm_result, regressor_name)`` — bar chart of t-statistics
  per channel for a given regressor.
- ``render_glm_summary(glm_result)`` — summary card with model metadata
  (number of regressors, channels, noise model, mean R² if available).

These are pure functions that return Plotly ``Figure`` or Dash component
layouts.  They carry no callbacks — embedding pages are responsible for
wiring interactivity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import html

from nirspy.domain.glm_result import GLMResult

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_glm_topo(
    glm_result: GLMResult,
    regressor_name: str,
    *,
    stat: str = "t_stat",
    significance_threshold: float = 0.05,
) -> go.Figure:
    """Render a bar chart of GLM statistics per channel for a regressor.

    Parameters
    ----------
    glm_result:
        Domain GLMResult object.
    regressor_name:
        Name of the regressor/condition to visualize.
    stat:
        Which statistic to plot: ``"t_stat"`` or ``"theta"`` (coefficient).
    significance_threshold:
        P-value threshold for highlighting significant channels.

    Returns
    -------
    go.Figure
        Plotly bar chart figure.

    Raises
    ------
    KeyError
        If *regressor_name* is not found in the result.
    ValueError
        If *stat* is not one of the supported values.
    """
    if stat not in ("t_stat", "theta"):
        msg = f"stat must be 't_stat' or 'theta', got {stat!r}"
        raise ValueError(msg)

    if regressor_name not in glm_result.regressor_names:
        raise KeyError(
            f"Regressor {regressor_name!r} not found. "
            f"Available: {glm_result.regressor_names}"
        )

    idx = glm_result.regressor_names.index(regressor_name)

    if stat == "t_stat":
        values = glm_result.t_stats[idx, :]
        y_label = "t-statistic"
    else:
        values = glm_result.theta[idx, :]
        y_label = "Coefficient (β)"

    p_values = glm_result.p_values[idx, :]
    channels = glm_result.channel_names

    # Color by significance
    colors = [
        "#2196F3" if p < significance_threshold else "#BDBDBD"
        for p in p_values
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=channels,
                y=values,
                marker_color=colors,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{y_label}: %{{y:.3f}}<br>"
                    "p-value: %{customdata:.4f}"
                    "<extra></extra>"
                ),
                customdata=p_values,
            )
        ]
    )

    fig.update_layout(
        title=f"GLM: {regressor_name} — {y_label} per channel",
        xaxis_title="Channel",
        yaxis_title=y_label,
        template="plotly_white",
        showlegend=False,
        height=400,
        margin={"t": 50, "b": 80, "l": 60, "r": 20},
    )

    # Add significance threshold annotation
    n_sig = int(np.sum(p_values < significance_threshold))
    fig.add_annotation(
        text=f"{n_sig}/{len(channels)} channels p < {significance_threshold}",
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.0,
        showarrow=False,
        font={"size": 11, "color": "#666"},
        xanchor="right",
    )

    return fig


def render_glm_summary(glm_result: GLMResult) -> html.Div:
    """Render a summary card with GLM model metadata.

    Parameters
    ----------
    glm_result:
        Domain GLMResult object.

    Returns
    -------
    html.Div
        Dash layout with summary statistics.
    """
    n_regressors = len(glm_result.regressor_names)
    n_channels = len(glm_result.channel_names)
    noise_model = glm_result.noise_model

    # Compute mean MSE as a proxy for goodness-of-fit
    mean_mse = float(np.mean(glm_result.mse))

    # R² from metadata if available
    r_squared: Any = glm_result.metadata.get("r_squared")

    summary_items: list[Any] = [
        _summary_row("Regressors", str(n_regressors)),
        _summary_row("Channels", str(n_channels)),
        _summary_row("Noise model", noise_model),
        _summary_row("Mean MSE", f"{mean_mse:.4e}"),
    ]

    if r_squared is not None:
        mean_r2 = float(np.mean(r_squared))
        summary_items.append(_summary_row("Mean R²", f"{mean_r2:.4f}"))

    summary_items.append(
        _summary_row(
            "Regressors list",
            ", ".join(glm_result.regressor_names),
        )
    )

    return html.Div(
        summary_items,
        style={
            "border": "1px solid #e0e0e0",
            "borderRadius": "8px",
            "padding": "16px",
            "backgroundColor": "#fafafa",
            "maxWidth": "500px",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summary_row(label: str, value: str) -> html.Div:
    """Create a single label-value row for the summary card."""
    return html.Div(
        [
            html.Span(
                f"{label}: ",
                style={"fontWeight": "bold", "color": "#333"},
            ),
            html.Span(value, style={"color": "#555"}),
        ],
        style={"marginBottom": "6px", "fontSize": "14px"},
    )
