"""GLM analysis block -- v0.4 (T-034).

GLMBlock: fits a first-level General Linear Model to haemodynamic data,
wrapping mne_nirs.statistics.run_glm via MNEAdapter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import mne.io

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError
from nirspy.engine.mne_adapter import MNEAdapter

# ---------------------------------------------------------------------------
# GLM Block
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

VALID_DRIFT_MODELS = ("cosine", "polynomial")
VALID_HRF_MODELS = (
    "glover",
    "spm",
    "fir",
    "glover + derivative",
    "glover + derivative + dispersion",
    "spm + derivative",
    "spm + derivative + dispersion",
)
VALID_NOISE_MODELS = ("ar1", "ols")


@dataclass(frozen=True)
class GLMParams:
    """Parameters for first-level GLM analysis.

    Attributes
    ----------
    event_id:
        Optional mapping of condition names to event codes.
        If None, all annotations are used as conditions.
    drift_model:
        Drift model for the design matrix. 'cosine' (default) or 'polynomial'.
    drift_high_freq:
        High-pass cutoff frequency (Hz) for cosine drift regressors. Default 0.01.
    hrf_model:
        Haemodynamic Response Function model. Default 'glover'.
    noise_model:
        Noise model for GLM estimation. 'ar1' (default) or 'ols'.
    condition_durations:
        Optional per-condition stimulus duration in seconds.
        Keys are condition names (matching annotation descriptions).
        When None, duration is read from raw annotations; falls back to 1.0 s.
    per_condition_groups:
        Optional grouping of conditions for the design matrix.
        Maps group label -> list of condition names to merge under that label.
        When None, each condition is modelled independently.
    """

    event_id: dict[str, int] | None = field(default=None)
    drift_model: str = "cosine"
    drift_high_freq: float = 0.01
    hrf_model: str = "glover"
    noise_model: str = "ar1"
    condition_durations: dict[str, float] | None = field(default=None)
    per_condition_groups: dict[str, list[str]] | None = field(default=None)


_GLM_SPEC = BlockSpec(
    block_id="glm",
    display_name="GLM (First-Level)",
    input_type=DataType.RAW_HAEMO,
    output_type=DataType.GLM_RESULT,
    params_class=GLMParams,
    description=(
        "Fits a first-level General Linear Model to haemodynamic data. "
        "Produces coefficients, t-statistics, and p-values per channel."
    ),
)


class GLMBlock:
    """First-level GLM analysis for fNIRS haemodynamic data.

    Pipeline:
    1. Extract stimulus events from raw annotations
    2. Build design matrix (HRF convolution + drift regressors)
    3. Fit GLM via mne_nirs.statistics.run_glm
    4. Return GLMResult with per-channel statistics

    Pipeline position: after Beer-Lambert and optional preprocessing
    (RAW_HAEMO input). Short-channel regression (T-033) is recommended
    upstream to remove systemic physiology before GLM fitting.

    References
    ----------
    Huppert et al., 2009; Tak & Ye, 2014; Luke et al., 2021 (mne-nirs).
    """

    SPEC: ClassVar[BlockSpec] = _GLM_SPEC

    def __init__(
        self,
        params: GLMParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: GLMParams = params or GLMParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _GLM_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute GLM analysis."""
        if not inputs:
            raise ValidationError(
                "GLMBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        # ---- GlobalConditions no-op (T-042) ----
        from nirspy.domain.conditions import resolve_conditions

        resolved = resolve_conditions(
            context.extra if hasattr(context, "extra") else {},
            {
                "condition_durations": getattr(
                    self.params, "condition_durations", None
                ),
                "per_condition_groups": getattr(
                    self.params, "per_condition_groups", None
                ),
            },
        )

        # Effective local params (may be overridden by GlobalConditions)
        eff_condition_durations: dict[str, float] | None = self.params.condition_durations
        eff_per_condition_groups: dict[str, list[str]] | None = self.params.per_condition_groups
        eff_event_id: dict[str, int] | None = self.params.event_id

        if resolved is not None:
            # Build condition_durations from ConditionConfig.duration
            eff_condition_durations = {
                name: cfg.duration
                for name, cfg in resolved.condition_configs.items()
            }

            # Build per_condition_groups from resolved.groups if present
            if resolved.groups:
                eff_per_condition_groups = {
                    grp.label: list(grp.conditions)
                    for grp in resolved.groups
                }
            else:
                eff_per_condition_groups = None

            # event_id is not altered — GLM relies on annotation descriptions
            # and condition_durations handles duration overrides.
            # included_occurrences selection is not supported at GLM level.

        # Validate params
        if self.params.drift_model not in VALID_DRIFT_MODELS:
            raise ValidationError(
                f"GLMBlock: drift_model must be one of {VALID_DRIFT_MODELS}, "
                f"got {self.params.drift_model!r}."
            )

        if self.params.hrf_model not in VALID_HRF_MODELS:
            raise ValidationError(
                f"GLMBlock: hrf_model must be one of {VALID_HRF_MODELS}, "
                f"got {self.params.hrf_model!r}."
            )

        if self.params.noise_model not in VALID_NOISE_MODELS:
            raise ValidationError(
                f"GLMBlock: noise_model must be one of {VALID_NOISE_MODELS}, "
                f"got {self.params.noise_model!r}."
            )

        if self.params.drift_high_freq <= 0:
            raise ValidationError(
                f"GLMBlock: drift_high_freq must be > 0, "
                f"got {self.params.drift_high_freq}."
            )

        if eff_condition_durations is not None:
            for cond, dur in eff_condition_durations.items():
                if not (
                    isinstance(dur, (int, float))
                    and math.isfinite(dur)
                    and dur > 0
                ):
                    raise ValidationError(
                        f"condition_durations[{cond!r}] must be a positive finite number, "
                        f"got {dur!r}"
                    )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # Apply GlobalConditions annotation filter (T-042) -- MUST happen before
        # events_from_annotations so that renamed labels (name vs original_name)
        # and excluded occurrences are visible to GLM event extraction.
        if resolved is not None:
            from nirspy.domain.conditions import GlobalConditions as _GC
            from nirspy.engine.mne_adapter import MNEAdapter as _Adapter

            gc: _GC = context.extra["global_conditions"]
            raw = _Adapter.filter_annotations_by_conditions(raw, gc)

        # Validate channel type -- must be haemoglobin data
        ch_types = set(raw.get_channel_types())
        if "hbo" not in ch_types and "hbr" not in ch_types:
            raise ValidationError(
                f"GLMBlock expects hbo/hbr channels (RAW_HAEMO), "
                f"got: {sorted(ch_types)}. "
                f"Ensure a BeerLambert block precedes this one."
            )

        glm_result = self._adapter.run_glm(
            raw,
            event_id=eff_event_id,
            drift_model=self.params.drift_model,
            high_pass=self.params.drift_high_freq,
            hrf_model=self.params.hrf_model,
            noise_model=self.params.noise_model,
            condition_durations=eff_condition_durations,
            per_condition_groups=eff_per_condition_groups,
        )

        metadata: dict[str, Any] = {
            "drift_model": self.params.drift_model,
            "drift_high_freq": self.params.drift_high_freq,
            "hrf_model": self.params.hrf_model,
            "noise_model": self.params.noise_model,
            "n_channels": len(glm_result.channel_names),
            "n_regressors": len(glm_result.regressor_names),
            "conditions": glm_result.metadata.get("conditions", []),
        }

        return BlockResult(
            data=glm_result,
            block_id=_GLM_SPEC.block_id,
            metadata=metadata,
        )
