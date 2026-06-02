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

from nirspy.blocks.analysis import ConditionGroup, ConditionWindow
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
    groups:
        Optional list of ConditionGroup definitions.  When set, the block
        uses ``MNEAdapter.create_epochs_per_group`` and returns a
        ``dict[str, mne.Epochs]`` keyed by group label.
        Mutually exclusive with ``per_condition_windows``.
    per_condition_windows:
        Optional per-condition temporal window overrides.  When set, the
        block uses ``MNEAdapter.create_epochs_per_condition`` and returns a
        ``dict[str, mne.Epochs]`` keyed by condition name.
        Mutually exclusive with ``groups``.
    """

    tmin: float = -0.5
    tmax: float = 5.0
    baseline_tmin: float | None = None
    baseline_tmax: float = 0.0
    reject_amplitude: float | None = None
    reject_gradient: float | None = None
    event_id: dict[str, int] | None = field(default=None)
    groups: list[ConditionGroup] | None = field(default=None)
    per_condition_windows: dict[str, ConditionWindow] | None = field(default=None)


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

    Output:
        - ``mne.Epochs`` — legacy path (no groups, no per_condition_windows)
        - ``dict[str, mne.Epochs]`` — when groups or per_condition_windows
          are specified (keyed by group label / condition name respectively)

    Both cases use DataType.EPOCHS as the declared output_type.
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

        # ---- GlobalConditions no-op (T-042) ----
        from nirspy.domain.conditions import resolve_conditions

        resolved = resolve_conditions(
            context.extra if hasattr(context, "extra") else {},
            {
                "per_condition_windows": getattr(
                    self.params, "per_condition_windows", None
                ),
                "groups": getattr(self.params, "groups", None),
                "event_id": getattr(self.params, "event_id", None),
            },
        )

        # Effective local params (may be overridden by GlobalConditions)
        eff_per_condition_windows: dict[str, ConditionWindow] | None = (
            self.params.per_condition_windows
        )
        eff_groups: list[ConditionGroup] | None = self.params.groups
        eff_event_id: dict[str, int] | None = self.params.event_id

        if resolved is not None:
            if resolved.groups:
                # Groups path: build list[ConditionGroup] from domain ConditionGroup
                eff_groups = [
                    ConditionGroup(
                        label=grp.label,
                        condition_names=list(grp.conditions),
                        tmin=grp.tmin,
                        tmax=grp.tmax,
                        baseline_tmin=grp.baseline_tmin,
                        baseline_tmax=grp.baseline_tmax,
                    )
                    for grp in resolved.groups
                ]
                eff_per_condition_windows = None
            else:
                # Per-condition-windows path: build from ConditionConfig
                eff_per_condition_windows = {
                    name: ConditionWindow(
                        tmin=cfg.tmin,
                        tmax=cfg.tmax,
                        baseline_tmin=cfg.baseline_tmin,
                        baseline_tmax=cfg.baseline_tmax,
                    )
                    for name, cfg in resolved.condition_configs.items()
                }
                eff_groups = None
            # event_id stays None — condition names from GlobalConditions are
            # used directly as epoch condition keys via per_condition_windows/groups.
            eff_event_id = None

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

        # Validate mutual exclusion of groups / per_condition_windows
        # (check against effective values which may have been set by GlobalConditions)
        if eff_groups is not None and eff_per_condition_windows is not None:
            raise ValidationError(
                "EpochsExtractionBlock: groups and per_condition_windows are "
                "mutually exclusive. Use one or the other, not both."
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

        # Build rejection dict
        reject: dict[str, float] | None = None
        if self.params.reject_amplitude is not None:
            reject = {
                "hbo": self.params.reject_amplitude,
                "hbr": self.params.reject_amplitude,
            }

        metadata: dict[str, Any]

        # ---- Per-groups path ----
        if eff_groups is not None:
            groups_dict: dict[str, ConditionGroup] = {
                grp.label: grp for grp in eff_groups
            }
            epochs_dict = self._adapter.create_epochs_per_group(
                raw,
                groups=groups_dict,
                reject=reject,
            )
            metadata = {
                "tmin": self.params.tmin,
                "tmax": self.params.tmax,
                "groups_used": True,
                "conditions": list(epochs_dict.keys()),
                "n_conditions": len(epochs_dict),
                "n_epochs_total": sum(
                    len(ep.events) for ep in epochs_dict.values()
                ),
            }
            for grp_label, ep in epochs_dict.items():
                metadata[f"n_epochs_{grp_label}"] = len(ep.events)

            return BlockResult(
                data=epochs_dict,
                block_id=_EPOCHS_SPEC.block_id,
                metadata=metadata,
            )

        # ---- Per-condition-windows path ----
        if eff_per_condition_windows is not None:
            # Resolve event_id
            used_event_id = (
                eff_event_id
                if eff_event_id is not None
                else event_id
            )
            default_window = (
                self.params.tmin,
                self.params.tmax,
                self.params.baseline_tmin if self.params.baseline_tmin is not None
                else self.params.tmin,
                self.params.baseline_tmax,
            )
            epochs_dict = self._adapter.create_epochs_per_condition(
                raw,
                used_event_id,
                default_window=default_window,
                per_condition_windows=eff_per_condition_windows,
                reject=reject,
            )
            metadata = {
                "tmin": self.params.tmin,
                "tmax": self.params.tmax,
                "per_condition_windows_used": True,
                "conditions": list(epochs_dict.keys()),
                "n_conditions": len(epochs_dict),
                "n_epochs_total": sum(
                    len(ep.events) for ep in epochs_dict.values()
                ),
            }
            for cond, ep in epochs_dict.items():
                metadata[f"n_epochs_{cond}"] = len(ep.events)

            return BlockResult(
                data=epochs_dict,
                block_id=_EPOCHS_SPEC.block_id,
                metadata=metadata,
            )

        # ---- Legacy single-Epochs path ----
        # Resolve event_id (use effective value which may be overridden by GlobalConditions)
        used_event_id = (
            eff_event_id if eff_event_id is not None else event_id
        )

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

        metadata = {
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
