"""Motion correction blocks -- v0.2 (T-015+).

TDDR (Temporal Derivative Distribution Repair) is the first motion
correction method, per Fishburn et al. 2019.  Parameter-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import mne.io

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError
from nirspy.engine.mne_adapter import MNEAdapter

# ---------------------------------------------------------------------------
# TDDR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TDDRParams:
    """TDDR is parameter-free -- this dataclass exists for registry consistency."""


_TDDR_SPEC = BlockSpec(
    block_id="tddr",
    display_name="TDDR Motion Correction",
    input_type=DataType.RAW_OD,
    output_type=DataType.RAW_OD,
    params_class=TDDRParams,
    description=(
        "Temporal Derivative Distribution Repair — removes motion "
        "artifacts (Fishburn et al., 2019)."
    ),
)


class TDDRBlock:
    """Remove motion artifacts via TDDR (Fishburn et al., 2019).

    Operates on optical density data.  Parameter-free — the algorithm
    repairs the temporal derivative distribution of the signal.
    """

    SPEC: ClassVar[BlockSpec] = _TDDR_SPEC

    def __init__(
        self,
        params: TDDRParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: TDDRParams = params or TDDRParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _TDDR_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute TDDR motion correction."""
        if not inputs:
            raise ValidationError(
                "TDDRBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # Validate channel type
        ch_types = set(raw.get_channel_types())
        if "fnirs_od" not in ch_types:
            raise ValidationError(
                f"TDDRBlock expects fnirs_od channels, "
                f"got: {sorted(ch_types)}. "
                f"Ensure an Optical Density block precedes this one."
            )

        result_raw = self._adapter.tddr(raw)

        return BlockResult(
            data=result_raw,
            block_id="tddr",
            metadata={"method": "tddr", "reference": "Fishburn et al., 2019"},
        )


# ---------------------------------------------------------------------------
# Spline Motion Correction (Scholkmann et al., 2010)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplineParams:
    """Parameters for spline interpolation motion correction.

    Attributes
    ----------
    threshold:
        Z-score cutoff for artifact detection on the temporal
        derivative.  Samples with |z| > threshold are flagged.
    spline_order:
        Order of the interpolating spline (1-5).  Default 3
        (cubic spline).
    """

    threshold: float = 3.0
    spline_order: int = 3


_SPLINE_SPEC = BlockSpec(
    block_id="spline_motion_correction",
    display_name="Spline Motion Correction",
    input_type=DataType.RAW_OD,
    output_type=DataType.RAW_OD,
    params_class=SplineParams,
    description=(
        "Spline interpolation motion correction — detects artifacts "
        "via z-score of the temporal derivative and interpolates "
        "with cubic spline (Scholkmann et al., 2010)."
    ),
)


class SplineBlock:
    """Remove motion artifacts via spline interpolation (Scholkmann et al., 2010).

    Operates on optical density data.  Detects artifacts by computing the
    z-score of the temporal derivative of each channel; samples exceeding
    *threshold* are interpolated with a spline of the given order and
    subtracted from the original signal.
    """

    SPEC: ClassVar[BlockSpec] = _SPLINE_SPEC

    def __init__(
        self,
        params: SplineParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: SplineParams = params or SplineParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _SPLINE_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute spline motion correction."""
        if not inputs:
            raise ValidationError(
                "SplineBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # Validate channel type
        ch_types = set(raw.get_channel_types())
        if "fnirs_od" not in ch_types:
            raise ValidationError(
                f"SplineBlock expects fnirs_od channels, "
                f"got: {sorted(ch_types)}. "
                f"Ensure an Optical Density block precedes this one."
            )

        result_raw = self._adapter.spline_motion_correction(
            raw,
            threshold=self.params.threshold,
            spline_order=self.params.spline_order,
        )

        return BlockResult(
            data=result_raw,
            block_id="spline_motion_correction",
            metadata={
                "method": "spline",
                "threshold": self.params.threshold,
                "spline_order": self.params.spline_order,
                "reference": "Scholkmann et al., 2010",
            },
        )
