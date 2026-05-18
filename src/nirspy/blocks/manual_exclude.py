"""ManualChannelExcludeBlock -- manually flag channels as bad.

Allows users to explicitly exclude channels from downstream analysis.
Channels are flagged in ``raw.info["bads"]`` (same as PruneChannelsBlock)
-- they are NOT removed, preserving reversibility.

Wavelength pairing: fNIRS channels come in wavelength pairs (e.g.
"S1_D1 760" and "S1_D1 850").  When the user selects a channel name
that is a single wavelength, this block expands it to both wavelengths
of the pair (same logic as PruneChannelsBlock hot-fix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import mne.io

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError


@dataclass(frozen=True)
class ManualChannelExcludeParams:
    """Parameters for manual channel exclusion."""

    channels: list[str] = field(default_factory=list)


_MCE_SPEC = BlockSpec(
    block_id="manual_channel_exclude",
    display_name="Manual Channel Exclude",
    input_type=DataType.ANY,
    output_type=DataType.ANY,
    params_class=ManualChannelExcludeParams,
    description="Manually flag channels as bad for downstream exclusion.",
)


class ManualChannelExcludeBlock:
    """Mark user-selected channels as bad."""

    SPEC: ClassVar[BlockSpec] = _MCE_SPEC

    def __init__(self, params: ManualChannelExcludeParams | None = None) -> None:
        self.params: ManualChannelExcludeParams = params or ManualChannelExcludeParams()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _MCE_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute manual channel exclusion."""
        if not inputs:
            raise ValidationError(
                "ManualChannelExcludeBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        if not self.params.channels:
            return BlockResult(
                data=raw,
                block_id=_MCE_SPEC.block_id,
                metadata={"excluded_channels": [], "n_excluded": 0},
            )

        all_ch_names = set(raw.ch_names)
        missing = [ch for ch in self.params.channels if ch not in all_ch_names]
        if missing:
            raise ValidationError(
                f"ManualChannelExcludeBlock: channels not found: "
                f"{missing}. Available: {sorted(all_ch_names)[:20]}..."
            )

        paired_bads: set[str] = set(self.params.channels)
        for ch_name in self.params.channels:
            prefix, sep, _ = ch_name.rpartition(" ")
            if not sep:
                continue
            for ch in raw.ch_names:
                if ch.startswith(prefix + " "):
                    paired_bads.add(ch)

        raw_copy = raw.copy()
        new_bads = list(set(raw_copy.info["bads"]) | paired_bads)
        raw_copy.info["bads"] = new_bads

        return BlockResult(
            data=raw_copy,
            block_id=_MCE_SPEC.block_id,
            metadata={
                "excluded_channels": sorted(paired_bads),
                "n_excluded": len(paired_bads),
            },
        )
