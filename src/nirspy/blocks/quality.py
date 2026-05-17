"""Quality control blocks -- Etapa 3 (T-004).

Two blocks for channel quality assessment:

1. ScalpCouplingIndexBlock -- computes SCI per channel (Pollonini et al., 2014)
2. PruneChannelsBlock -- marks channels with low SCI as bads
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
# ScalpCouplingIndex
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalpCouplingIndexParams:
    """No configurable parameters -- MNE computes SCI without user params."""


_SCI_SPEC = BlockSpec(
    block_id="scalp_coupling_index",
    display_name="Scalp Coupling Index (SCI)",
    input_type=DataType.RAW_OD,
    output_type=DataType.RAW_OD,
    params_class=ScalpCouplingIndexParams,
    description="Computes SCI per channel. Stores values in metadata for pruning.",
)


class ScalpCouplingIndexBlock:
    """Compute Scalp Coupling Index per channel (Pollonini et al., 2014).

    SCI measures the correlation between adjacent wavelengths at each
    source-detector pair. Values in [0, 1]; higher = better coupling.

    Invariant: input Raw must have fnirs_od channel type.
    Output: same Raw (unchanged), SCI values stored in BlockResult.metadata["sci_values"].
    """

    SPEC: ClassVar[BlockSpec] = _SCI_SPEC

    def __init__(
        self,
        params: ScalpCouplingIndexParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: ScalpCouplingIndexParams = params or ScalpCouplingIndexParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _SCI_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute SCI computation."""
        if not inputs:
            raise ValidationError(
                "ScalpCouplingIndexBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # Validate channel type
        ch_types = set(raw.get_channel_types())
        if "fnirs_od" not in ch_types:
            raise ValidationError(
                f"ScalpCouplingIndexBlock expects fnirs_od channels, "
                f"got: {sorted(ch_types)}. "
                f"Ensure an OpticalDensity block precedes this one."
            )

        sci_values = self._adapter.scalp_coupling_index(raw)

        return BlockResult(
            data=raw,
            block_id=_SCI_SPEC.block_id,
            metadata={
                "sci_values": sci_values,
                "n_channels": len(raw.ch_names),
            },
        )


# ---------------------------------------------------------------------------
# PruneChannels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PruneChannelsParams:
    """Parameters for channel pruning based on SCI.

    Attributes
    ----------
    sci_threshold:
        Channels with SCI below this value are marked as bads.
        Default 0.5 per Pollonini et al. (2014).
    """

    sci_threshold: float = 0.5


_PRUNE_SPEC = BlockSpec(
    block_id="prune_channels",
    display_name="Prune Channels",
    input_type=DataType.RAW_OD,
    output_type=DataType.RAW_OD,
    params_class=PruneChannelsParams,
    description="Marks channels with SCI below threshold as bads.",
)


class PruneChannelsBlock:
    """Mark channels with poor scalp coupling as bads.

    Reads SCI values from the metadata of the preceding block (typically
    ScalpCouplingIndexBlock). Channels with SCI below the threshold are
    added to raw.info["bads"] -- they are NOT removed, preserving reversibility.

    Invariant: requires metadata["sci_values"] from preceding block.
    """

    SPEC: ClassVar[BlockSpec] = _PRUNE_SPEC

    def __init__(
        self,
        params: PruneChannelsParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: PruneChannelsParams = params or PruneChannelsParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _PRUNE_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute channel pruning based on SCI threshold."""
        if not inputs:
            raise ValidationError(
                "PruneChannelsBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # PruneChannels needs SCI values from the execution context.
        # In the linear pipeline, the runner passes BlockResult.data as input.
        # SCI values are propagated via the metadata mechanism -- the runner
        # stores metadata from the previous block in context.extra["prev_metadata"].
        # However, to keep the block self-contained and not couple to runner
        # internals, we also check raw.info for SCI annotation.
        #
        # Strategy: check context.extra for sci_values first, then raise.
        sci_values: dict[str, float] | None = None

        if hasattr(context, "extra") and isinstance(context.extra, dict):
            sci_values = context.extra.get("sci_values")

        if sci_values is None:
            raise ValidationError(
                "PruneChannelsBlock requires SCI values from the preceding "
                "ScalpCouplingIndex block. Ensure ScalpCouplingIndex runs "
                "immediately before PruneChannels in the pipeline."
            )

        if self.params.sci_threshold < 0 or self.params.sci_threshold > 1:
            raise ValidationError(
                f"PruneChannelsBlock: sci_threshold must be in [0, 1], "
                f"got {self.params.sci_threshold}."
            )

        # Mark channels below threshold as bads
        raw_copy = raw.copy()
        bads: list[str] = []
        for ch_name, sci_val in sci_values.items():
            if sci_val < self.params.sci_threshold and ch_name in raw_copy.ch_names:
                bads.append(ch_name)

        # Extend existing bads (don't overwrite)
        raw_copy.info["bads"] = list(set(raw_copy.info["bads"] + bads))

        return BlockResult(
            data=raw_copy,
            block_id=_PRUNE_SPEC.block_id,
            metadata={
                "pruned_channels": bads,
                "n_pruned": len(bads),
                "sci_threshold": self.params.sci_threshold,
                "sci_values": sci_values,
            },
        )
