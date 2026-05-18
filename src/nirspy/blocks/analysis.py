"""Analysis blocks -- Etapa 3 (T-004).

BlockAverageBlock: computes epoch-averaged HRF per stimulus condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import mne
import mne.io

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError
from nirspy.engine.mne_adapter import MNEAdapter

# ---------------------------------------------------------------------------
# BlockAverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockAverageParams:
    """Parameters for block averaging (HRF computation).

    Attributes
    ----------
    tmin:
        Epoch start relative to event onset (seconds). Default -2.0.
    tmax:
        Epoch end relative to event onset (seconds). Default 18.0.
    baseline_tmin:
        Baseline correction start. Default -2.0.
    baseline_tmax:
        Baseline correction end. Default 0.0.
    reject_by_amplitude:
        Whether to reject epochs exceeding amplitude threshold.
    amplitude_threshold:
        Rejection threshold in mol/L (default 80e-6 = 80 uM).
    pick_conditions:
        List of condition names to include. None = all conditions.
    """

    tmin: float = -2.0
    tmax: float = 18.0
    baseline_tmin: float = -2.0
    baseline_tmax: float = 0.0
    reject_by_amplitude: bool = True
    amplitude_threshold: float = 80e-6
    pick_conditions: list[str] | None = field(default=None)


_BA_SPEC = BlockSpec(
    block_id="block_average",
    display_name="Block Average (HRF)",
    input_type=DataType.RAW_HAEMO,
    output_type=DataType.EVOKED,
    params_class=BlockAverageParams,
    description="Computes epoch-averaged HRF per stimulus condition.",
)


class BlockAverageBlock:
    """Compute epoch-averaged HRF per stimulus condition.

    Pipeline:
    1. Extract events from raw annotations
    2. Create epochs (tmin/tmax window, baseline correction)
    3. Optionally reject epochs by amplitude threshold
    4. Average per condition -> dict[str, Evoked]

    Invariant: input Raw must have hbo/hbr channels (RAW_HAEMO).
    Raw must have annotations/events; raises ValidationError if none found.
    """

    SPEC: ClassVar[BlockSpec] = _BA_SPEC

    def __init__(
        self,
        params: BlockAverageParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: BlockAverageParams = params or BlockAverageParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _BA_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute block averaging."""
        if not inputs:
            raise ValidationError(
                "BlockAverageBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        # Validate params
        if self.params.baseline_tmin > self.params.baseline_tmax:
            raise ValidationError(
                f"BlockAverageBlock: baseline_tmin ({self.params.baseline_tmin}) "
                f"must be <= baseline_tmax ({self.params.baseline_tmax})."
            )

        if self.params.tmin >= self.params.tmax:
            raise ValidationError(
                f"BlockAverageBlock: tmin ({self.params.tmin}) "
                f"must be < tmax ({self.params.tmax})."
            )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # Validate channel type
        ch_types = set(raw.get_channel_types())
        if "hbo" not in ch_types and "hbr" not in ch_types:
            raise ValidationError(
                f"BlockAverageBlock expects hbo/hbr channels (RAW_HAEMO), "
                f"got: {sorted(ch_types)}. "
                f"Ensure a BeerLambert block precedes this one."
            )

        # Check for annotations/events
        events_from_annot, event_id = mne.events_from_annotations(
            raw, verbose=False
        )
        if len(events_from_annot) == 0:
            raise ValidationError(
                "BlockAverageBlock: no events found in raw annotations. "
                "Ensure the SNIRF file contains stimulus annotations."
            )

        # Filter conditions if pick_conditions is set
        used_event_id: dict[str, int] | None = None
        if self.params.pick_conditions is not None:
            used_event_id = {
                k: v for k, v in event_id.items()
                if k in self.params.pick_conditions
            }
            if not used_event_id:
                raise ValidationError(
                    f"BlockAverageBlock: none of pick_conditions "
                    f"{self.params.pick_conditions} found in event_id "
                    f"{list(event_id.keys())}."
                )
        else:
            used_event_id = event_id

        # Build rejection dict
        reject: dict[str, float] | None = None
        if self.params.reject_by_amplitude:
            reject = {
                "hbo": self.params.amplitude_threshold,
                "hbr": self.params.amplitude_threshold,
            }

        # Create epochs via adapter
        epochs = self._adapter.create_epochs(
            raw,
            tmin=self.params.tmin,
            tmax=self.params.tmax,
            baseline_tmin=self.params.baseline_tmin,
            baseline_tmax=self.params.baseline_tmax,
            reject=reject,
            event_id=used_event_id,
        )

        # Average per condition via adapter
        evoked_dict = self._adapter.average_epochs(epochs)

        # Build metadata with epoch stats
        skipped_conditions = [
            cond for cond in epochs.event_id if cond not in evoked_dict
        ]
        metadata: dict[str, Any] = {
            "conditions": list(evoked_dict.keys()),
            "n_conditions": len(evoked_dict),
            "n_epochs_total": len(epochs.events),
            "skipped_conditions": skipped_conditions,
            "tmin": self.params.tmin,
            "tmax": self.params.tmax,
        }

        # Add per-condition epoch counts
        for condition in evoked_dict:
            metadata[f"n_epochs_{condition}"] = len(epochs[condition])

        if epochs.drop_log is not None:
            n_dropped = sum(
                1 for log in epochs.drop_log if len(log) > 0
            )
            metadata["n_epochs_dropped"] = n_dropped

        return BlockResult(
            data=evoked_dict,
            block_id=_BA_SPEC.block_id,
            metadata=metadata,
        )
