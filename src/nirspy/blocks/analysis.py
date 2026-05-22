"""Analysis blocks -- Etapa 3 (T-004).

BlockAverageBlock: computes epoch-averaged HRF per stimulus condition.
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
# BlockAverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionWindow:
    """Temporal window override for a single condition.

    Each field mirrors the corresponding global parameter in
    :class:`BlockAverageParams`.  When a condition has an entry in
    ``per_condition_windows`` these values take precedence over the
    globals for that condition only.
    """

    tmin: float
    tmax: float
    baseline_tmin: float
    baseline_tmax: float


def _validate_condition_window(name: str, window: ConditionWindow) -> None:
    """Raise :class:`ValidationError` if *window* has invalid ranges."""
    if window.tmin >= window.tmax:
        raise ValidationError(
            f"ConditionWindow {name!r}: tmin ({window.tmin}) "
            f"must be < tmax ({window.tmax})."
        )
    if window.baseline_tmin > window.baseline_tmax:
        raise ValidationError(
            f"ConditionWindow {name!r}: baseline_tmin "
            f"({window.baseline_tmin}) must be <= baseline_tmax "
            f"({window.baseline_tmax})."
        )


@dataclass(frozen=True)
class ConditionGroup:
    """A named group of SNIRF conditions sharing temporal parameters.

    Used by ``BlockAverageParams.per_condition_groups`` to let users
    aggregate multiple SNIRF condition keys under a custom label with
    shared tmin/tmax/baseline windows. The label becomes the key in
    the resulting ``dict[str, Evoked]``.
    """

    label: str
    condition_names: list[str]
    tmin: float
    tmax: float
    baseline_tmin: float
    baseline_tmax: float


def _validate_condition_group(name: str, group: ConditionGroup) -> None:
    """Raise :class:`ValidationError` if *group* has invalid ranges."""
    if group.tmin >= group.tmax:
        raise ValidationError(
            f"ConditionGroup {name!r}: tmin ({group.tmin}) "
            f"must be < tmax ({group.tmax})."
        )
    if group.baseline_tmin > group.baseline_tmax:
        raise ValidationError(
            f"ConditionGroup {name!r}: baseline_tmin "
            f"({group.baseline_tmin}) must be <= baseline_tmax "
            f"({group.baseline_tmax})."
        )
    if not group.condition_names:
        raise ValidationError(
            f"ConditionGroup {name!r}: condition_names must not be empty."
        )


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
    per_condition_windows:
        Per-condition temporal window overrides.  Empty dict (default)
        uses the global window for every condition.
    """

    tmin: float = -2.0
    tmax: float = 18.0
    baseline_tmin: float = -2.0
    baseline_tmax: float = 0.0
    reject_by_amplitude: bool = True
    amplitude_threshold: float = 80e-6
    pick_conditions: list[str] | None = field(default=None)
    per_condition_windows: dict[str, ConditionWindow] = field(
        default_factory=dict,
    )
    per_condition_groups: dict[str, ConditionGroup] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        """Coerce raw dicts to ConditionWindow for YAML round-trip.

        Partial dicts (e.g. only ``tmin`` set) are allowed: any missing field
        falls back to the corresponding global parameter so the semantics are
        "override only what you specify, inherit the rest from the block-level
        defaults".
        """
        if self.per_condition_windows:
            coerced: dict[str, ConditionWindow] = {}
            for key, val in self.per_condition_windows.items():
                if isinstance(val, dict):
                    defaults_used = {
                        "tmin", "tmax", "baseline_tmin", "baseline_tmax",
                    } - set(val.keys())
                    merged = {
                        "tmin": self.tmin,
                        "tmax": self.tmax,
                        "baseline_tmin": self.baseline_tmin,
                        "baseline_tmax": self.baseline_tmax,
                        **val,
                    }
                    if defaults_used:
                        logger.warning(
                            "per_condition_windows[%r]: filled missing "
                            "field(s) %s from global defaults.",
                            key,
                            sorted(defaults_used),
                        )
                    coerced[key] = ConditionWindow(**merged)
                else:
                    coerced[key] = val
            object.__setattr__(self, "per_condition_windows", coerced)

        # Mutual exclusion: per_condition_windows OR per_condition_groups (D3)
        if self.per_condition_windows and self.per_condition_groups:
            raise ValidationError(
                "BlockAverageParams: per_condition_windows and "
                "per_condition_groups are mutually exclusive (D3). "
                "Use one or the other, not both."
            )

        # Coerce raw dicts to ConditionGroup for YAML round-trip
        if self.per_condition_groups:
            coerced_groups: dict[str, ConditionGroup] = {}
            for grp_key, grp_val in self.per_condition_groups.items():
                if isinstance(grp_val, dict):
                    coerced_groups[grp_key] = ConditionGroup(**grp_val)
                else:
                    coerced_groups[grp_key] = grp_val
            object.__setattr__(self, "per_condition_groups", coerced_groups)


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

    When ``per_condition_windows`` is non-empty the block creates one
    :class:`mne.Epochs` per condition (each with its own window) via
    :meth:`MNEAdapter.create_epochs_per_condition`.

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
        required = {
            "tmin": self.params.tmin,
            "tmax": self.params.tmax,
            "baseline_tmin": self.params.baseline_tmin,
            "baseline_tmax": self.params.baseline_tmax,
        }
        missing = [name for name, val in required.items() if val is None]
        if missing:
            raise ValidationError(
                f"BlockAverageBlock: required parameter(s) "
                f"{missing} must not be empty."
            )

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


        # Validate each per-condition window
        for cond_name, window in self.params.per_condition_windows.items():
            _validate_condition_window(cond_name, window)

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

        # Validate per_condition_windows keys exist in event_id
        if self.params.per_condition_windows:
            unknown = (
                set(self.params.per_condition_windows) - set(used_event_id)
            )
            if unknown:
                raise ValidationError(
                    f"BlockAverageBlock: per_condition_windows contains "
                    f"condition(s) {sorted(unknown)} not found in "
                    f"event_id {sorted(used_event_id.keys())}."
                )

        # Build rejection dict
        reject: dict[str, float] | None = None
        if self.params.reject_by_amplitude:
            reject = {
                "hbo": self.params.amplitude_threshold,
                "hbr": self.params.amplitude_threshold,
            }

        # ---- Per-condition-groups path (T-024) ----
        if self.params.per_condition_groups:
            # Validate each group
            for grp_name, grp in self.params.per_condition_groups.items():
                _validate_condition_group(grp_name, grp)

            epochs_dict = self._adapter.create_epochs_per_group(
                raw,
                groups=self.params.per_condition_groups,
                reject=reject,
            )
            evoked_dict = self._adapter.average_epochs(epochs_dict)

            metadata: dict[str, Any] = {
                "conditions": list(evoked_dict.keys()),
                "n_conditions": len(evoked_dict),
                "n_epochs_total": sum(
                    len(ep.events) for ep in epochs_dict.values()
                ),
                "per_condition_groups_used": True,
                "tmin": self.params.tmin,
                "tmax": self.params.tmax,
            }
            for grp_label, ep in epochs_dict.items():
                metadata[f"n_epochs_{grp_label}"] = len(ep.events)

        # ---- Per-condition path vs legacy single-Epochs path ----
        elif self.params.per_condition_windows:
            default_window = (
                self.params.tmin,
                self.params.tmax,
                self.params.baseline_tmin,
                self.params.baseline_tmax,
            )
            epochs_dict = self._adapter.create_epochs_per_condition(
                raw,
                used_event_id,
                default_window=default_window,
                per_condition_windows=self.params.per_condition_windows,
                reject=reject,
            )
            evoked_dict = self._adapter.average_epochs(epochs_dict)

            # Metadata for per-condition path
            windows_used: dict[str, dict[str, float]] = {}
            n_epochs_total = 0
            skipped_conditions: list[str] = []
            metadata = {}

            for cond in used_event_id:
                if cond in self.params.per_condition_windows:
                    w = self.params.per_condition_windows[cond]
                    windows_used[cond] = {
                        "tmin": w.tmin,
                        "tmax": w.tmax,
                        "baseline_tmin": w.baseline_tmin,
                        "baseline_tmax": w.baseline_tmax,
                    }
                else:
                    windows_used[cond] = {
                        "tmin": self.params.tmin,
                        "tmax": self.params.tmax,
                        "baseline_tmin": self.params.baseline_tmin,
                        "baseline_tmax": self.params.baseline_tmax,
                    }
                if cond in epochs_dict:
                    n_ep = len(epochs_dict[cond].events)
                    n_epochs_total += n_ep
                    metadata[f"n_epochs_{cond}"] = n_ep
                if cond not in evoked_dict:
                    skipped_conditions.append(cond)

            metadata.update({
                "conditions": list(evoked_dict.keys()),
                "n_conditions": len(evoked_dict),
                "n_epochs_total": n_epochs_total,
                "skipped_conditions": skipped_conditions,
                "tmin": self.params.tmin,
                "tmax": self.params.tmax,
                "per_condition_used": True,
                "windows_used": windows_used,
            })

        else:
            # Legacy single-Epochs path
            epochs = self._adapter.create_epochs(
                raw,
                tmin=self.params.tmin,
                tmax=self.params.tmax,
                baseline_tmin=self.params.baseline_tmin,
                baseline_tmax=self.params.baseline_tmax,
                reject=reject,
                event_id=used_event_id,
            )
            evoked_dict = self._adapter.average_epochs(epochs)

            skipped_conditions = [
                cond for cond in epochs.event_id
                if cond not in evoked_dict
            ]
            metadata = {
                "conditions": list(evoked_dict.keys()),
                "n_conditions": len(evoked_dict),
                "n_epochs_total": len(epochs.events),
                "skipped_conditions": skipped_conditions,
                "tmin": self.params.tmin,
                "tmax": self.params.tmax,
            }

            for condition in evoked_dict:
                metadata[f"n_epochs_{condition}"] = len(
                    epochs[condition]
                )

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
