"""Preprocessing blocks -- Etapa 2 (T-003).

Three blocks that transform raw fNIRS data:

1. OpticalDensityBlock -- intensity -> OD (log ratio)
2. BeerLambertBlock -- OD -> HbO/HbR concentrations
3. BandpassFilterBlock -- IIR/FIR bandpass filter (ADR-012: ANY/ANY)
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
# OpticalDensity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpticalDensityParams:
    """No configurable parameters -- conversion is deterministic."""


_OD_SPEC = BlockSpec(
    block_id="optical_density",
    display_name="Optical Density",
    input_type=DataType.RAW,
    output_type=DataType.RAW_OD,
    params_class=OpticalDensityParams,
    description="Converts raw intensity to optical density (log ratio).",
)


class OpticalDensityBlock:
    """Convert raw CW amplitude to optical density via MNE optical_density().

    Invariant: input Raw must have fnirs_cw_amplitude channel type.
    """

    SPEC: ClassVar[BlockSpec] = _OD_SPEC

    def __init__(
        self,
        params: OpticalDensityParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: OpticalDensityParams = params or OpticalDensityParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _OD_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute optical density conversion."""
        if not inputs:
            raise ValidationError(
                "OpticalDensityBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # Validate channel type
        ch_types = set(raw.get_channel_types())
        if "fnirs_cw_amplitude" not in ch_types:
            raise ValidationError(
                f"OpticalDensityBlock expects fnirs_cw_amplitude channels, "
                f"got: {sorted(ch_types)}. "
                f"Ensure a LoadSnirf block precedes this one in the pipeline."
            )

        result_raw = self._adapter.raw_to_od(raw)

        return BlockResult(
            data=result_raw,
            block_id=_OD_SPEC.block_id,
            metadata={"n_channels": len(result_raw.ch_names)},
        )

# ---------------------------------------------------------------------------
# BeerLambert
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeerLambertParams:
    """Parameters for modified Beer-Lambert Law conversion.

    Attributes
    ----------
    ppf:
        Partial Pathlength Factor. Default 6.0 per Yucel et al. (2021).
    """

    ppf: float = 6.0


_BL_SPEC = BlockSpec(
    block_id="beer_lambert",
    display_name="Modified Beer-Lambert Law",
    input_type=DataType.RAW_OD,
    output_type=DataType.RAW_HAEMO,
    params_class=BeerLambertParams,
    description="Converts OD to HbO/HbR concentrations via mBLL.",
)


class BeerLambertBlock:
    """Convert optical density to haemoglobin concentrations via mBLL.

    Invariant: input Raw must have fnirs_od channel type. PPF must be > 0.
    """

    SPEC: ClassVar[BlockSpec] = _BL_SPEC

    def __init__(
        self,
        params: BeerLambertParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: BeerLambertParams = params or BeerLambertParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _BL_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute Beer-Lambert conversion."""
        if not inputs:
            raise ValidationError(
                "BeerLambertBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        if self.params.ppf <= 0:
            raise ValidationError(
                f"BeerLambertBlock: ppf must be > 0, got {self.params.ppf}."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        ch_types = set(raw.get_channel_types())
        if "fnirs_od" not in ch_types:
            raise ValidationError(
                f"BeerLambertBlock expects fnirs_od channels, "
                f"got: {sorted(ch_types)}. "
                f"Ensure an OpticalDensity block precedes this one."
            )

        result_raw = self._adapter.beer_lambert(raw, ppf=self.params.ppf)

        return BlockResult(
            data=result_raw,
            block_id=_BL_SPEC.block_id,
            metadata={
                "n_channels": len(result_raw.ch_names),
                "ppf": self.params.ppf,
            },
        )

# ---------------------------------------------------------------------------
# BandpassFilter (ADR-012: ANY/ANY, ADR-013: 0.01-0.5 Hz default)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandpassFilterParams:
    """Parameters for bandpass filter.

    Attributes
    ----------
    l_freq:
        Low cutoff frequency in Hz. None disables highpass component.
    h_freq:
        High cutoff frequency in Hz. None disables lowpass component.
    method:
        iir (default) or fir.
    iir_params:
        Optional dict passed to MNE raw.filter(iir_params=...).
    """

    l_freq: float | None = 0.01
    h_freq: float | None = 0.5
    method: str = "iir"
    iir_params: dict[str, Any] | None = None


_BP_SPEC = BlockSpec(
    block_id="bandpass_filter",
    display_name="Bandpass Filter",
    input_type=DataType.ANY,
    output_type=DataType.ANY,
    params_class=BandpassFilterParams,
    description="IIR/FIR bandpass filter via MNE Raw.filter().",
)


class BandpassFilterBlock:
    """Apply bandpass filter to any fNIRS Raw data.

    Uses DataType.ANY for both input and output (ADR-012) because filtering
    does not change the semantic type of the data.

    Invariant: at least one of l_freq / h_freq must be not None.
    When both are set, l_freq < h_freq.
    """

    SPEC: ClassVar[BlockSpec] = _BP_SPEC

    def __init__(
        self,
        params: BandpassFilterParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: BandpassFilterParams = params or BandpassFilterParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _BP_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute bandpass filter."""
        if not inputs:
            raise ValidationError(
                "BandpassFilterBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        # Validate params
        if self.params.l_freq is None and self.params.h_freq is None:
            raise ValidationError(
                "BandpassFilterBlock: at least one of l_freq or h_freq must be set."
            )

        if (
            self.params.l_freq is not None
            and self.params.h_freq is not None
            and self.params.l_freq >= self.params.h_freq
        ):
            raise ValidationError(
                f"BandpassFilterBlock: l_freq ({self.params.l_freq}) must be "
                f"< h_freq ({self.params.h_freq})."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        result_raw = self._adapter.bandpass_filter(
            raw,
            l_freq=self.params.l_freq,
            h_freq=self.params.h_freq,
            method=self.params.method,
            iir_params=self.params.iir_params,
        )

        return BlockResult(
            data=result_raw,
            block_id=_BP_SPEC.block_id,
            metadata={
                "l_freq": self.params.l_freq,
                "h_freq": self.params.h_freq,
                "method": self.params.method,
            },
        )
