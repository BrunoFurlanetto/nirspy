"""Tooltip registry -- scientific context for each pipeline block.

Provides short descriptions and literature references for every
registered block type, rendered as ``dbc.Tooltip`` attached to
catalog items or block cards.

References follow Best Practices for fNIRS (Yuecel et al., 2021)
and MNE-NIRS documentation where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass

import dash_bootstrap_components as dbc


@dataclass(frozen=True)
class _TooltipEntry:
    """Internal registry entry for a block tooltip."""

    description: str
    reference: str


_TOOLTIP_REGISTRY: dict[str, _TooltipEntry] = {
    "load_snirf": _TooltipEntry(
        description=(
            "Load a SNIRF file and return raw fNIRS data. "
            "SNIRF is the standard format for fNIRS data exchange."
        ),
        reference="SNIRF specification (https://github.com/fNIRS/snirf)",
    ),
    "optical_density": _TooltipEntry(
        description=(
            "Convert raw intensity to optical density (OD). "
            "Required before Beer-Lambert conversion."
        ),
        reference="MNE-NIRS: mne.preprocessing.nirs.optical_density",
    ),
    "beer_lambert": _TooltipEntry(
        description=(
            "Apply the modified Beer-Lambert Law to convert OD to "
            "haemoglobin concentrations (HbO/HbR)."
        ),
        reference=(
            "Kocsis et al., 2006; "
            "MNE-NIRS: mne.preprocessing.nirs.beer_lambert_law"
        ),
    ),
    "bandpass_filter": _TooltipEntry(
        description=(
            "Apply a bandpass filter to remove physiological noise "
            "(cardiac ~1 Hz, respiration ~0.3 Hz) and slow drift."
        ),
        reference=(
            "Yuecel et al., 2021 -- Best Practices for fNIRS, "
            "Section 3.3: Bandpass Filtering"
        ),
    ),
    "scalp_coupling_index": _TooltipEntry(
        description=(
            "Compute Scalp Coupling Index (SCI) per channel. "
            "SCI measures signal quality via cross-correlation "
            "between wavelength pairs. Values close to 1 indicate "
            "good optode-scalp contact."
        ),
        reference="Pollonini et al., 2014; Yuecel et al., 2021 Section 3.1",
    ),
    "prune_channels": _TooltipEntry(
        description=(
            "Mark channels with low SCI as bad. "
            "Channels are flagged, not removed, preserving reversibility."
        ),
        reference="Pollonini et al., 2014; Yuecel et al., 2021 Section 3.1",
    ),
    "block_average": _TooltipEntry(
        description=(
            "Epoch the data around events and compute the average "
            "haemodynamic response function (HRF) per condition."
        ),
        reference=(
            "Yuecel et al., 2021 -- Best Practices for fNIRS, "
            "Section 4: Statistical Analysis"
        ),
    ),
    "tddr": _TooltipEntry(
        description=(
            "Temporal Derivative Distribution Repair (TDDR). "
            "Parameter-free method that removes motion artifacts by "
            "repairing the temporal derivative distribution of the signal. "
            "Applied to optical density data."
        ),
        reference="Fishburn et al., 2019; MNE-NIRS: temporal_derivative_distribution_repair",
    ),
    "spline_motion_correction": _TooltipEntry(
        description=(
            "Spline interpolation motion correction (Scholkmann et al., 2010). "
            "Detects artifacts via z-score of the temporal derivative and "
            "interpolates affected windows with cubic spline. "
            "Parameters: threshold (z-score cutoff, default 3.0), "
            "spline_order (default 3)."
        ),
        reference="Scholkmann et al., 2010",
    ),
    "wavelet_motion_correction": _TooltipEntry(
        description=(
            "Wavelet-based motion correction (Molavi & Dumont, 2012). "
            "Decomposes signal via Discrete Wavelet Transform (DWT), "
            "soft-thresholds detail coefficients based on IQR, "
            "and reconstructs. Parameters: wavelet family (default sym8), "
            "IQR multiplier (default 1.5)."
        ),
        reference=(
            "Molavi & Dumont, 2012, Physiological Measurement"
        ),
    ),
    "manual_channel_exclude": _TooltipEntry(
        description=(
            "Manually flag channels as bad. Pipeline already pre-processes "
            "with Scalp Coupling Index — check the QC tab first. "
            "Use this block only if you want to disable channels manually "
            "beyond (or instead of) the automatic prune. "
            "Channels with good SCI can be kept."
        ),
        reference="User-driven quality control",
    ),
    "short_channel_regression": _TooltipEntry(
        description=(
            "Regress out superficial-layer noise using short-separation "
            "channels (<8 mm). Short channels capture systemic physiology "
            "(heart rate, respiration) that contaminates long channels. "
            "Improves specificity of the haemodynamic response estimate."
        ),
        reference="Tachtsidis & Scholkmann, 2016; Yuecel et al., 2021 Section 3.4",
    ),
    "epochs_extraction": _TooltipEntry(
        description=(
            "Extract epochs time-locked to stimulus events. "
            "Segments the continuous signal around event onsets with "
            "configurable pre- and post-stimulus windows, baseline correction, "
            "and optional amplitude-based rejection."
        ),
        reference="MNE: mne.Epochs; Yuecel et al., 2021 Section 4",
    ),
    "export_table": _TooltipEntry(
        description=(
            "Export a DataFrame result (e.g., GLM summary) to disk as "
            "CSV or Parquet. Useful for downstream statistical analysis "
            "in R, SPSS, or Python notebooks."
        ),
        reference="nirspy pipeline sink — no MNE reference",
    ),
    "glm": _TooltipEntry(
        description=(
            "First-level General Linear Model (GLM) analysis. "
            "Fits a design matrix with condition regressors convolved with "
            "the haemodynamic response function (HRF) to estimate "
            "channel-level t-statistics and beta weights per condition."
        ),
        reference="Friston et al., 1994; mne_nirs.statistics.run_glm",
    ),
    "html_report": _TooltipEntry(
        description=(
            "Generate a standalone HTML report from pipeline results. "
            "Embeds interactive Plotly figures and summary tables. "
            "No external dependencies — the output file is self-contained."
        ),
        reference="nirspy pipeline sink — no MNE reference",
    ),
}


def tooltip_for(block_id: str) -> dbc.Tooltip | None:
    """Return a ``dbc.Tooltip`` for the given block, or *None* if unknown.

    The tooltip targets the catalog item with matching ``block_id``
    in its pattern-matching ID.

    Parameters
    ----------
    block_id:
        Registry block identifier (e.g. ``"optical_density"``).

    Returns
    -------
    dbc.Tooltip | None
        Tooltip component, or ``None`` if no entry is registered.
    """
    entry = _TOOLTIP_REGISTRY.get(block_id)
    if entry is None:
        return None

    text = f"{entry.description}\n\nRef: {entry.reference}"

    return dbc.Tooltip(
        text,
        target={"type": "catalog-item", "block_id": block_id},
        placement="right",
        delay={"show": 300, "hide": 100},
    )
