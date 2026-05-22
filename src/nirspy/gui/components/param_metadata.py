"""Parameter metadata registry -- scientific labels, units and ranges.

Provides ``ParamMeta`` entries keyed by ``(block_id, field_name)`` so the
:mod:`~nirspy.gui.components.param_editor` can render rich labels, tooltips
with references, and HTML5 min/max/step validation on numeric inputs.

ParamMeta is **presentation-only metadata** -- it never alters block behaviour
or pipeline serialisation (ADR-007).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamMeta:
    """Display metadata for a single block parameter field.

    Attributes
    ----------
    label:
        Human-readable name shown instead of the raw field name.
    unit:
        Physical unit string (``"Hz"``, ``"s"``, ``"μM"``).  ``None`` when
        the parameter is dimensionless or non-numeric.
    description:
        Short tooltip body explaining the parameter.
    reference:
        Optional literature citation or URL.
    min:
        Minimum valid value (HTML5 hint -- does not block submission).
    max:
        Maximum valid value.
    step:
        Increment step for the numeric spinner.
    choices:
        If set, the field is rendered as a dropdown instead of a text input.
    """

    label: str
    unit: str | None = None
    description: str = ""
    reference: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None


# ---------------------------------------------------------------------------
# Registry keyed by (block_id, field_name)
# ---------------------------------------------------------------------------

_PARAM_META_REGISTRY: dict[tuple[str, str], ParamMeta] = {
    # -- BeerLambert ---------------------------------------------------------
    ("beer_lambert", "ppf"): ParamMeta(
        label="DPF (pathlength factor)",
        description=(
            "Differential Pathlength Factor. Varies with age and wavelength. "
            "Adult ~6.0; ≥50 years ~6.6 (Scholkmann & Wolf 2013; "
            "Duncan 1996). Edit according to your study protocol."
        ),
        reference="Scholkmann & Wolf, 2013; Duncan et al., 1996",
        min=3,
        max=8,
        step=0.01,
    ),
    # -- BandpassFilter ------------------------------------------------------
    ("bandpass_filter", "l_freq"): ParamMeta(
        label="Low cutoff",
        unit="Hz",
        description="High-pass corner frequency. Removes slow drift.",
        min=0,
        max=1,
        step=0.001,
    ),
    ("bandpass_filter", "h_freq"): ParamMeta(
        label="High cutoff",
        unit="Hz",
        description="Low-pass corner frequency. Removes cardiac/respiration noise.",
        min=0,
        max=5,
        step=0.01,
    ),
    # -- PruneChannels -------------------------------------------------------
    ("prune_channels", "sci_threshold"): ParamMeta(
        label="SCI threshold",
        description=(
            "Channels with Scalp Coupling Index below this value are "
            "flagged as bad."
        ),
        reference="Pollonini et al., 2014",
        min=0,
        max=1,
        step=0.01,
    ),
    # -- BlockAverage --------------------------------------------------------
    ("block_average", "tmin"): ParamMeta(
        label="Window start",
        unit="s",
        description=(
            "Epoch start relative to event onset. "
            "Typical task paradigms: -5 to -30 s pre-stimulus."
        ),
        min=-60,
        max=0,
        step=1.0,
    ),
    ("block_average", "tmax"): ParamMeta(
        label="Window end",
        unit="s",
        description=(
            "Epoch end relative to event onset. "
            "Real task durations: 20-60 s (task) + 20-30 s (baseline)."
        ),
        min=0,
        max=180,
        step=1.0,
    ),
    ("block_average", "baseline_tmin"): ParamMeta(
        label="Baseline start",
        unit="s",
        description=(
            "Start of the baseline correction window. "
            "Typical: -20 to -30 s for long tasks."
        ),
        min=-60,
        max=0,
        step=1.0,
    ),
    ("block_average", "baseline_tmax"): ParamMeta(
        label="Baseline end",
        unit="s",
        description=(
            "End of the baseline correction window. "
            "Usually 0 s (onset) or slightly positive."
        ),
        min=-60,
        max=10,
        step=1.0,
    ),
    ("block_average", "amplitude_threshold"): ParamMeta(
        label="Amplitude reject",
        unit="M (mol/L)",
        description="Epochs exceeding this peak-to-peak threshold are rejected.",
        min=0,
        step=1e-6,
    ),
    # -- LoadSnirf -----------------------------------------------------------
    ("load_snirf", "path"): ParamMeta(
        label="SNIRF path",
        description="Filesystem path to the .snirf input file.",
    ),
    # -- SplineMotionCorrection -----------------------------------------------
    ("spline_motion_correction", "threshold"): ParamMeta(
        label="Artifact threshold (z-score)",
        description=(
            "Samples whose temporal-derivative z-score exceeds this "
            "value are flagged as motion artifacts."
        ),
        reference="Scholkmann et al., 2010",
        min=1.0,
        max=10.0,
        step=0.5,
    ),
    ("spline_motion_correction", "spline_order"): ParamMeta(
        label="Spline order",
        description=(
            "Order of the interpolating spline. "
            "1 = linear, 3 = cubic (recommended), 5 = quintic."
        ),
        reference="Scholkmann et al., 2010",
        min=1,
        max=5,
        step=1,
    ),
    # -- WaveletMotionCorrection ------------------------------------------------
    ("wavelet_motion_correction", "wavelet"): ParamMeta(
        label="Wavelet family",
        description=(
            "Mother wavelet for the DWT decomposition. "
            "sym8 is recommended for fNIRS motion correction."
        ),
        reference="Molavi & Dumont, 2012",
        choices=["sym8", "sym4", "sym6", "db4", "db6", "haar"],
    ),
    ("wavelet_motion_correction", "iqr_multiplier"): ParamMeta(
        label="IQR multiplier",
        description=(
            "Multiplier for the interquartile range (IQR) used to "
            "compute the soft-threshold. Lower values = more aggressive "
            "correction."
        ),
        reference="Molavi & Dumont, 2012",
        min=0.5,
        max=5.0,
        step=0.1,
    ),
    # -- ManualChannelExclude ------------------------------------------------
    ("manual_channel_exclude", "channels"): ParamMeta(
        label="Channels to exclude",
        description=(
            "Select channels to manually flag as bad. "
            "Flagged channels are excluded from downstream analysis."
        ),
    ),
}


def metadata_for(block_id: str, field_name: str) -> ParamMeta | None:
    """Look up display metadata for a block parameter.

    Returns ``None`` when no entry is registered -- callers should fall
    back to the raw field name.
    """
    return _PARAM_META_REGISTRY.get((block_id, field_name))
