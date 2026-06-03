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

    Modes (D8)
    ----------
    Exactly one of ``condition_names`` or ``event_indices`` must be
    non-empty.  Using both simultaneously is forbidden — ``__post_init__``
    raises :class:`~nirspy.domain.exceptions.ValidationError`.

    condition_names:
        Classic mode (T-024): groups all occurrences of the listed
        SNIRF condition keys together.
    event_indices:
        Timeline mode (T-030): groups specific occurrences identified by
        their chronological index in ``raw.annotations`` (sorted by onset).
        Index 0 = first occurrence across *all* stim annotations.
    """

    label: str
    condition_names: list[str] = field(default_factory=list)
    tmin: float = -2.0
    tmax: float = 18.0
    baseline_tmin: float = -2.0
    baseline_tmax: float = 0.0
    event_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Enforce mutual exclusion between condition_names and event_indices."""
        has_names = bool(self.condition_names)
        has_indices = bool(self.event_indices)
        if has_names and has_indices:
            raise ValidationError(
                f"ConditionGroup {self.label!r}: condition_names and "
                "event_indices are mutually exclusive (D8). "
                "Populate one or the other, not both."
            )
        if not has_names and not has_indices:
            raise ValidationError(
                f"ConditionGroup {self.label!r}: either condition_names or "
                "event_indices must be non-empty."
            )


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

        # Coerce raw dicts to ConditionGroup for YAML round-trip.
        # event_indices defaults to [] when absent so legacy YAML (T-024,
        # condition_names only) continues to deserialise without changes.
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

        # ---- GlobalConditions no-op (T-042) ----
        # When global conditions are present in context.extra the local
        # condition params (pick_conditions, per_condition_windows,
        # per_condition_groups) are overridden by the resolved values.
        from nirspy.domain.conditions import resolve_conditions

        resolved = resolve_conditions(
            context.extra if hasattr(context, "extra") else {},
            {
                "pick_conditions": getattr(self.params, "pick_conditions", None),
                "per_condition_windows": getattr(
                    self.params, "per_condition_windows", None
                ) or None,
                "per_condition_groups": getattr(
                    self.params, "per_condition_groups", None
                ) or None,
            },
        )

        # Effective local params (may be overridden by GlobalConditions)
        eff_pick_conditions: list[str] | None = self.params.pick_conditions
        eff_per_condition_windows: dict[str, ConditionWindow] = dict(
            self.params.per_condition_windows
        )
        eff_per_condition_groups: dict[str, ConditionGroup] = dict(
            self.params.per_condition_groups
        )

        if resolved is not None:
            # Build effective pick_conditions from condition names
            eff_pick_conditions = list(resolved.condition_configs.keys())

            # Build per_condition_windows from ConditionConfig tmin/tmax/baseline
            eff_per_condition_windows = {
                name: ConditionWindow(
                    tmin=cfg.tmin,
                    tmax=cfg.tmax,
                    baseline_tmin=cfg.baseline_tmin,
                    baseline_tmax=cfg.baseline_tmax,
                )
                for name, cfg in resolved.condition_configs.items()
            }

            # Build per_condition_groups from resolved.groups if present
            if resolved.groups:
                eff_per_condition_groups = {
                    grp.label: ConditionGroup(
                        label=grp.label,
                        condition_names=list(grp.conditions),
                        tmin=grp.tmin,
                        tmax=grp.tmax,
                        baseline_tmin=grp.baseline_tmin,
                        baseline_tmax=grp.baseline_tmax,
                    )
                    for grp in resolved.groups
                }
                # Groups path takes precedence — clear per_condition_windows
                eff_per_condition_windows = {}
            else:
                eff_per_condition_groups = {}

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
        for cond_name, window in eff_per_condition_windows.items():
            _validate_condition_window(cond_name, window)

        # ---- T-042 debug: log effective condition params ----
        _src = "global_conditions" if resolved is not None else "local_params"
        if eff_per_condition_groups:
            for _grp_name, _grp in eff_per_condition_groups.items():
                logger.debug(
                    "BlockAverageBlock [T-042] group=%r  conditions=%s  "
                    "tmin=%.3f  tmax=%.3f  source=%s",
                    _grp_name,
                    list(_grp.condition_names),
                    _grp.tmin,
                    _grp.tmax,
                    _src,
                )
        elif eff_per_condition_windows:
            for _cond, _win in eff_per_condition_windows.items():
                logger.debug(
                    "BlockAverageBlock [T-042] condition=%r  tmin=%.3f  "
                    "tmax=%.3f  baseline=[%.3f, %.3f]  source=%s",
                    _cond,
                    _win.tmin,
                    _win.tmax,
                    _win.baseline_tmin,
                    _win.baseline_tmax,
                    _src,
                )
        else:
            logger.debug(
                "BlockAverageBlock [T-042] pick_conditions=%s  global "
                "tmin=%.3f  tmax=%.3f  source=%s",
                eff_pick_conditions,
                self.params.tmin,
                self.params.tmax,
                _src,
            )

        # Log duration specifically when coming from global_conditions
        if resolved is not None:
            for _cond, _cfg in resolved.condition_configs.items():
                logger.debug(
                    "BlockAverageBlock [T-042] condition=%r  duration=%.4f s  "
                    "(stim duration from GlobalConditions — not used by "
                    "BlockAverage directly, only by GLM)",
                    _cond,
                    _cfg.duration,
                )

        raw: mne.io.BaseRaw = next(iter(inputs.values()))

        # Apply GlobalConditions annotation filter (T-042) -- MUST happen before
        # events_from_annotations so that renamed labels (name vs original_name)
        # and excluded occurrences are visible to epoch creation.
        if resolved is not None:
            from nirspy.domain.conditions import GlobalConditions as _GC
            from nirspy.engine.mne_adapter import MNEAdapter as _Adapter

            gc: _GC = context.extra["global_conditions"]
            raw = _Adapter.filter_annotations_by_conditions(raw, gc)

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
        if eff_pick_conditions is not None:
            used_event_id = {
                k: v for k, v in event_id.items()
                if k in eff_pick_conditions
            }
            if not used_event_id:
                raise ValidationError(
                    f"BlockAverageBlock: none of pick_conditions "
                    f"{eff_pick_conditions} found in event_id "
                    f"{list(event_id.keys())}."
                )
        else:
            used_event_id = event_id

        # Filter per_condition_windows to only keys present in event_id.
        # Stale keys can appear when the user swaps the SNIRF file while
        # per-condition windows are already configured. Raising would give
        # a confusing error; silently discarding (with warning) is correct
        # because the GUI sync callback (sync_conditions_on_path_change)
        # already removed them in the GUI path -- this is defence-in-depth
        # for YAML-loaded pipelines or any other path that bypasses the GUI.
        # Restores T-012 hotfix (2d7b63d) regressed by T-024 (#37).
        filtered_pcw: dict[str, ConditionWindow] = dict(eff_per_condition_windows)
        if filtered_pcw:
            unknown = set(filtered_pcw) - set(used_event_id)
            if unknown:
                import warnings

                warnings.warn(
                    f"BlockAverageBlock: per_condition_windows contains "
                    f"condition(s) {sorted(unknown)} not found in the current "
                    f"SNIRF event_id {sorted(used_event_id.keys())}. "
                    f"These entries will be skipped (stale keys from a "
                    f"previous SNIRF file).",
                    UserWarning,
                    stacklevel=2,
                )
                filtered_pcw = {
                    k: v for k, v in filtered_pcw.items() if k in used_event_id
                }

        # Build rejection dict
        reject: dict[str, float] | None = None
        if self.params.reject_by_amplitude:
            reject = {
                "hbo": self.params.amplitude_threshold,
                "hbr": self.params.amplitude_threshold,
            }

        # ---- Per-condition-groups path (T-024) ----
        if eff_per_condition_groups:
            # Validate each group
            for grp_name, grp in eff_per_condition_groups.items():
                _validate_condition_group(grp_name, grp)

            epochs_dict = self._adapter.create_epochs_per_group(
                raw,
                groups=eff_per_condition_groups,
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
        elif filtered_pcw:
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
                per_condition_windows=filtered_pcw,
                reject=reject,
            )
            evoked_dict = self._adapter.average_epochs(epochs_dict)

            # Metadata for per-condition path
            windows_used: dict[str, dict[str, float]] = {}
            n_epochs_total = 0
            skipped_conditions: list[str] = []
            metadata = {}

            for cond in used_event_id:
                if cond in filtered_pcw:
                    w = filtered_pcw[cond]
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
