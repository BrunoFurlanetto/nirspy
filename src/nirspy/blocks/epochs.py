"""Epochs extraction block -- v0.4 (T-035).

EpochsExtractionBlock: segments continuous haemodynamic data into
time-locked epochs around stimulus events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import mne
import mne.io

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError
from nirspy.engine.mne_adapter import MNEAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EpochsExtraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochsExtractionParams:
    """Parameters for epoch extraction.

    Attributes
    ----------
    tmin:
        Epoch start relative to event onset (seconds). Default -0.5.
    tmax:
        Epoch end relative to event onset (seconds). Default 5.0.
    baseline_tmin:
        Baseline correction start (seconds). None disables baseline start
        bound. Default None.
    baseline_tmax:
        Baseline correction end (seconds). Default 0.0.
    reject_amplitude:
        Amplitude rejection threshold in mol/L. None disables rejection.
    reject_gradient:
        Gradient rejection threshold. None disables gradient rejection.
        (Reserved for future use -- MNE flat/reject_tmin/reject_tmax.)
    event_id:
        Optional mapping of condition names to event codes.
        None = use all annotation-derived events.
    """

    tmin: float = -0.5
    tmax: float = 5.0
    baseline_tmin: float | None = None
    baseline_tmax: float = 0.0
    reject_amplitude: float | None = None
    reject_gradient: float | None = None
    event_id: dict[str, int] | None = field(default=None)


_EPOCHS_SPEC = BlockSpec(
    block_id="epochs_extraction",
    display_name="Epochs Extraction",
    input_type=DataType.RAW_HAEMO,
    output_type=DataType.EPOCHS,
    params_class=EpochsExtractionParams,
    description=(
        "Segments continuous haemodynamic data into time-locked epochs "
        "around stimulus events."
    ),
)


class EpochsExtractionBlock:
    """Extract epochs from continuous haemodynamic data.

    Segments RAW_HAEMO into time-locked epochs around stimulus annotations.
    Supports optional amplitude-based epoch rejection and baseline correction.

    Pipeline position: after Beer-Lambert and optionally after filtering/
    short-channel regression.

    Output: mne.Epochs object (DataType.EPOCHS).
    """

    SPEC: ClassVar[BlockSpec] = _EPOCHS_SPEC

    def __init__(
        self,
        params: EpochsExtractionParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: EpochsExtractionParams = params or EpochsExtractionParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _EPOCHS_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute epochs extraction."""
        if not inputs:
            raise ValidationError(
                "EpochsExtractionBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # Validate channel type -- must be haemoglobin data
        ch_types = set(raw.get_channel_types())
        if "hbo" not in ch_types and "hbr" not in ch_types:
            raise ValidationError(
                f"EpochsExtractionBlock expects hbo/hbr channels "
                f"(RAW_HAEMO), got: {sorted(ch_types)}. "
                f"Ensure a BeerLambert block precedes this one."
            )

        # Validate temporal parameters
        if self.params.tmin >= self.params.tmax:
            raise ValidationError(
                f"EpochsExtractionBlock: tmin ({self.params.tmin}) "
                f"must be < tmax ({self.params.tmax})."
            )

        if (
            self.params.baseline_tmin is not None
            and self.params.baseline_tmin > self.params.baseline_tmax
        ):
            raise ValidationError(
                f"EpochsExtractionBlock: baseline_tmin "
                f"({self.params.baseline_tmin}) must be <= "
                f"baseline_tmax ({self.params.baseline_tmax})."
            )

        # Check for annotations/events
        events_from_annot, event_id = mne.events_from_annotations(
            raw, verbose=False
        )
        if len(events_from_annot) == 0:
            raise ValidationError(
                "EpochsExtractionBlock: no events found in raw annotations. "
                "Ensure the SNIRF file contains stimulus annotations."
            )

        # Resolve event_id
        used_event_id = (
            self.params.event_id if self.params.event_id is not None else event_id
        )

        # Build rejection dict
        reject: dict[str, float] | None = None
        if self.params.reject_amplitude is not None:
            reject = {
                "hbo": self.params.reject_amplitude,
                "hbr": self.params.reject_amplitude,
            }

        # Build baseline tuple
        baseline: tuple[float | None, float] = (
            self.params.baseline_tmin,
            self.params.baseline_tmax,
        )

        # Create epochs using the adapter
        epochs = self._adapter.create_epochs(
            raw,
            tmin=self.params.tmin,
            tmax=self.params.tmax,
            baseline_tmin=baseline[0] if baseline[0] is not None else self.params.tmin,
            baseline_tmax=baseline[1],
            reject=reject,
            event_id=used_event_id,
        )

        # Compute metadata
        n_epochs_total = len(epochs.events)
        n_epochs_dropped = 0
        drop_log_summary: dict[str, int] = {}

        if epochs.drop_log is not None:
            for log_entry in epochs.drop_log:
                if len(log_entry) > 0:
                    n_epochs_dropped += 1
                    for reason in log_entry:
                        drop_log_summary[reason] = (
                            drop_log_summary.get(reason, 0) + 1
                        )

        metadata: dict[str, Any] = {
            "n_epochs_total": n_epochs_total,
            "n_epochs_dropped": n_epochs_dropped,
            "drop_log_summary": drop_log_summary,
            "tmin": self.params.tmin,
            "tmax": self.params.tmax,
            "baseline": baseline,
            "conditions": list(epochs.event_id.keys()),
            "n_conditions": len(epochs.event_id),
        }

        return BlockResult(
            data=epochs,
            block_id=_EPOCHS_SPEC.block_id,
            metadata=metadata,
        )
