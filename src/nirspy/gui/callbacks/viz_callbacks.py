"""Visualization callbacks -- update plots when run-results changes."""

from __future__ import annotations

from typing import Any

from dash import Input, Output, callback, dcc, html

from nirspy.domain.glm_result import GLMResult
from nirspy.gui.callbacks.execution_callbacks import _VIZ_CACHE
from nirspy.gui.components.condition_selector import (
    render_condition_selector,
)
from nirspy.gui.components.glm_topo import render_glm_summary, render_glm_topo
from nirspy.gui.components.hrf_plot import render_hrf_plot
from nirspy.gui.components.probe_viewer import render_probe_viewer
from nirspy.gui.components.qc_dashboard import render_qc_dashboard
from nirspy.gui.components.raw_data_plot import render_raw_data_plot


def _get_cached_results(
    run_results: dict[str, Any] | None,
) -> list[Any] | None:
    """Retrieve MNE objects from the viz cache."""
    if not run_results or "cache_key" not in run_results:
        return None
    cache_key: str = run_results["cache_key"]
    entry = _VIZ_CACHE.get(cache_key)
    if entry is None:
        return None
    return entry.get("results")  # type: ignore[no-any-return]


def _find_last_raw(results: list[Any]) -> Any:
    """Find the last Raw-like result in the execution chain."""
    try:
        import mne
    except ImportError:
        return None
    for r in reversed(results):
        if isinstance(r.data, mne.io.BaseRaw):
            return r.data
    return None


def _find_evoked_dict(results: list[Any]) -> dict[str, Any] | None:
    """Find the evoked dict from BlockAverage results."""
    try:
        import mne
    except ImportError:
        return None
    for r in reversed(results):
        if isinstance(r.data, dict) and any(
            isinstance(v, mne.Evoked) for v in r.data.values()
        ):
            return r.data
    return None


def _find_sci_values(results: list[Any]) -> Any:
    """Extract SCI values from block metadata.

    ScalpCouplingIndexBlock stores ``sci_values`` as ``dict[str, float]``
    keyed by channel name. Returns the list of float values in dict order.
    """
    for r in results:
        sci = r.metadata.get("sci_values")
        if sci is None:
            continue
        if isinstance(sci, dict):
            return list(sci.values())
        return sci
    return None


def _find_sci_ch_names(results: list[Any]) -> list[str] | None:
    """Extract channel names associated with SCI values."""
    for r in results:
        sci = r.metadata.get("sci_values")
        if isinstance(sci, dict):
            return list(sci.keys())
        ch_names = r.metadata.get("sci_ch_names")
        if ch_names is not None:
            return ch_names  # type: ignore[no-any-return]
    # Fallback: try to get from Raw
    try:
        import mne
    except ImportError:
        return None
    for r in results:
        if isinstance(r.data, mne.io.BaseRaw):
            return list(r.data.ch_names)
    return None


def _find_glm_result(results: list[Any]) -> GLMResult | None:
    """Find the most recent GLMResult in the execution chain."""
    for r in reversed(results):
        if isinstance(r.data, GLMResult):
            return r.data
    return None


@callback(
    Output("raw-data-plot-container", "children"),
    Input("run-results", "data"),
)
def update_raw_plot(
    run_results: dict[str, Any] | None,
) -> Any:
    """Render the raw data plot when results are available."""
    results = _get_cached_results(run_results)
    if results is None:
        return render_raw_data_plot(None)
    raw = _find_last_raw(results)
    return render_raw_data_plot(raw)


@callback(
    Output("probe-viewer-container", "children"),
    Input("run-results", "data"),
)
def update_probe(
    run_results: dict[str, Any] | None,
) -> Any:
    """Render the probe viewer when results are available."""
    results = _get_cached_results(run_results)
    if results is None:
        return render_probe_viewer(None)
    raw = _find_last_raw(results)
    if raw is None:
        return render_probe_viewer(None)
    bads = list(raw.info.get("bads", []))
    return render_probe_viewer(raw.info, bads=bads)


@callback(
    Output("qc-dashboard-container", "children"),
    Input("run-results", "data"),
)
def update_qc(
    run_results: dict[str, Any] | None,
) -> Any:
    """Render the QC dashboard when SCI data is available."""
    results = _get_cached_results(run_results)
    if results is None:
        return render_qc_dashboard(None)
    sci = _find_sci_values(results)
    ch_names = _find_sci_ch_names(results)
    return render_qc_dashboard(sci, ch_names)


@callback(
    Output("condition-selector-container", "children"),
    Input("run-results", "data"),
)
def update_conditions(
    run_results: dict[str, Any] | None,
) -> Any:
    """Populate the condition selector from evoked results."""
    results = _get_cached_results(run_results)
    if results is None:
        return render_condition_selector(None)
    evoked_dict = _find_evoked_dict(results)
    if evoked_dict is None:
        return render_condition_selector(None)
    return render_condition_selector(list(evoked_dict.keys()))


@callback(
    Output("hrf-plot-container", "children"),
    Input("run-results", "data"),
    Input("condition-selector", "value"),
    Input("hrf-discard-toggle", "value"),
    Input("hrf-discard-tmin", "value"),
    Input("hrf-discard-tmax", "value"),
)
def update_hrf(
    run_results: dict[str, Any] | None,
    selected_conditions: list[str] | None,
    discard_toggle: bool | None,
    discard_tmin: float | None,
    discard_tmax: float | None,
) -> Any:
    """Render the HRF plot filtered by selected conditions.

    The *discard region* overlay (``discard_toggle``, ``discard_tmin``,
    ``discard_tmax``) is purely visual and does not affect any
    computation.  Invalid ranges (tmin >= tmax) are silently ignored.
    """
    results = _get_cached_results(run_results)
    if results is None:
        return render_hrf_plot(None)
    evoked_dict = _find_evoked_dict(results)
    return render_hrf_plot(
        evoked_dict,
        selected_conditions,
        discard_toggle=bool(discard_toggle),
        discard_tmin=discard_tmin,
        discard_tmax=discard_tmax,
    )


@callback(
    Output("glm-regressor-selector", "options"),
    Output("glm-regressor-selector", "value"),
    Output("glm-regressor-selector", "disabled"),
    Output("glm-summary-container", "children"),
    Input("run-results", "data"),
)
def update_glm_tab(run_results: dict[str, Any] | None) -> tuple[Any, ...]:
    """Populate GLM dropdown and summary card when results are available."""
    results = _get_cached_results(run_results)
    if results is None:
        return [], None, True, ""
    glm = _find_glm_result(results)
    if glm is None:
        return [], None, True, ""
    options = [{"label": r, "value": r} for r in glm.regressor_names]
    first = glm.regressor_names[0] if glm.regressor_names else None
    return options, first, False, render_glm_summary(glm)


@callback(
    Output("glm-topo-container", "children"),
    Input("run-results", "data"),
    Input("glm-regressor-selector", "value"),
)
def update_glm_topo(
    run_results: dict[str, Any] | None,
    selected_regressor: str | None,
) -> Any:
    """Render GLM bar chart for the selected regressor."""
    results = _get_cached_results(run_results)
    if results is None or selected_regressor is None:
        return html.P(
            "Run pipeline with GLM block to see results.",
            className="text-muted text-center py-4",
        )
    glm = _find_glm_result(results)
    if glm is None or selected_regressor not in glm.regressor_names:
        return html.P(
            "No GLM results available.",
            className="text-muted text-center py-4",
        )
    return dcc.Graph(figure=render_glm_topo(glm, selected_regressor))
