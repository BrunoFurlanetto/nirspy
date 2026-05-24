"""Signal enhancement blocks -- v0.4 (T-033).

ShortChannelRegressionBlock: regresses out short-channel signals
(scalp physiology) from long channels, improving cortical sensitivity.
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
# ShortChannelRegression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShortChannelRegressionParams:
    """Parameters for short-channel regression.

    Attributes
    ----------
    max_dist:
        Maximum source-detector distance (in meters) to consider a channel
        as "short". Default 0.015 m (15 mm) per fNIRS best practices.
    """

    max_dist: float = 0.015


_SCR_SPEC = BlockSpec(
    block_id="short_channel_regression",
    display_name="Short Channel Regression",
    input_type=DataType.RAW_HAEMO,
    output_type=DataType.RAW_HAEMO,
    params_class=ShortChannelRegressionParams,
    description=(
        "Regresses out short-channel (scalp) signals from long channels "
        "to improve cortical sensitivity."
    ),
)


class ShortChannelRegressionBlock:
    """Regress out short-channel signals from long channels.

    Short-separation channels (source-detector distance < max_dist) capture
    systemic physiology (cardiac, respiration, Mayer waves) without cortical
    contribution. Regressing these from long channels isolates the cortical
    haemodynamic response.

    Pipeline position: after Beer-Lambert (RAW_HAEMO).

    References
    ----------
    Saager & Berger, 2005; Brigadoi & Cooper, 2015.
    """

    SPEC: ClassVar[BlockSpec] = _SCR_SPEC

    def __init__(
        self,
        params: ShortChannelRegressionParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: ShortChannelRegressionParams = (
            params or ShortChannelRegressionParams()
        )
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _SCR_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute short-channel regression."""
        if not inputs:
            raise ValidationError(
                "ShortChannelRegressionBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # Validate channel type -- must be haemoglobin data
        ch_types = set(raw.get_channel_types())
        if "hbo" not in ch_types and "hbr" not in ch_types:
            raise ValidationError(
                f"ShortChannelRegressionBlock expects hbo/hbr channels "
                f"(RAW_HAEMO), got: {sorted(ch_types)}. "
                f"Ensure a BeerLambert block precedes this one."
            )

        # Validate max_dist parameter
        if self.params.max_dist <= 0:
            raise ValidationError(
                f"ShortChannelRegressionBlock: max_dist must be > 0, "
                f"got {self.params.max_dist}."
            )

        result_raw = self._adapter.short_channel_regression(
            raw, max_dist=self.params.max_dist
        )

        # Compute metadata -- identify short channels by distance
        import numpy as np

        short_ch_names: list[str] = []
        for ch in raw.info["chs"]:
            src = ch["loc"][3:6]
            det = ch["loc"][6:9]
            dist = float(np.linalg.norm(src - det))
            if dist <= self.params.max_dist:
                short_ch_names.append(ch["ch_name"])

        metadata: dict[str, Any] = {
            "max_dist": self.params.max_dist,
            "n_short_channels": len(short_ch_names),
            "short_channels": short_ch_names,
            "n_long_channels": len(raw.ch_names) - len(short_ch_names),
        }

        return BlockResult(
            data=result_raw,
            block_id=_SCR_SPEC.block_id,
            metadata=metadata,
        )
