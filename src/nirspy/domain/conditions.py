from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionConfig:
    name: str
    original_name: str
    included_occurrences: tuple[int, ...] | None = None
    duration: float = 1.0
    tmin: float = -2.0
    tmax: float = 18.0
    baseline_tmin: float = -2.0
    baseline_tmax: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.original_name:
            raise ValueError("original_name must not be empty")
        if not (self.duration > 0 and math.isfinite(self.duration)):
            raise ValueError(
                f"duration must be > 0 and finite, got {self.duration}"
            )
        if self.tmin >= self.tmax:
            raise ValueError(f"tmin ({self.tmin}) must be less than tmax ({self.tmax})")
        if self.baseline_tmin > self.baseline_tmax:
            raise ValueError(
                f"baseline_tmin ({self.baseline_tmin}) must be less than or equal to "
                f"baseline_tmax ({self.baseline_tmax})"
            )


@dataclass(frozen=True)
class ConditionGroup:
    label: str
    conditions: tuple[str, ...]
    tmin: float = -2.0
    tmax: float = 18.0
    baseline_tmin: float = -2.0
    baseline_tmax: float = 0.0

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("label must not be empty")
        if not self.conditions:
            raise ValueError("conditions must not be empty")
        if self.tmin >= self.tmax:
            raise ValueError(f"tmin ({self.tmin}) must be less than tmax ({self.tmax})")
        if self.baseline_tmin > self.baseline_tmax:
            raise ValueError(
                f"baseline_tmin ({self.baseline_tmin}) must be less than or equal to "
                f"baseline_tmax ({self.baseline_tmax})"
            )


@dataclass(frozen=True)
class GlobalConditions:
    conditions: tuple[ConditionConfig, ...]
    groups: tuple[ConditionGroup, ...] | None = None

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("conditions must not be empty")
        names = [c.name for c in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("ConditionConfig names must be unique")
        if self.groups:
            labels = [g.label for g in self.groups]
            if len(labels) != len(set(labels)):
                raise ValueError("ConditionGroup labels must be unique")
            valid_names = set(names)
            for g in self.groups:
                for cname in g.conditions:
                    if cname not in valid_names:
                        raise ValueError(
                            f"Group {g.label!r} references unknown condition {cname!r}"
                        )


@dataclass(frozen=True)
class ResolvedConditions:
    condition_configs: dict[str, ConditionConfig]
    groups: tuple[ConditionGroup, ...] | None
    source: str  # "global" | "local"


def resolve_conditions(
    context_extra: dict[str, Any],
    local_condition_params: dict[str, Any] | None = None,
) -> ResolvedConditions | None:
    gc: GlobalConditions | None = context_extra.get("global_conditions")
    if gc is None:
        return None
    if local_condition_params:
        non_empty = {k: v for k, v in local_condition_params.items() if v}
        if non_empty:
            logger.warning(
                "Global conditions active — local params %s ignored.",
                list(non_empty.keys()),
            )
    return ResolvedConditions(
        condition_configs={c.name: c for c in gc.conditions},
        groups=gc.groups,
        source="global",
    )


def global_conditions_to_dict(gc: GlobalConditions) -> dict[str, Any]:
    def cond_to_dict(c: ConditionConfig) -> dict[str, Any]:
        return {
            "name": c.name,
            "original_name": c.original_name,
            "included_occurrences": (
                list(c.included_occurrences)
                if c.included_occurrences is not None
                else None
            ),
            "duration": c.duration,
            "tmin": c.tmin,
            "tmax": c.tmax,
            "baseline_tmin": c.baseline_tmin,
            "baseline_tmax": c.baseline_tmax,
        }

    def group_to_dict(g: ConditionGroup) -> dict[str, Any]:
        return {
            "label": g.label,
            "conditions": list(g.conditions),
            "tmin": g.tmin,
            "tmax": g.tmax,
            "baseline_tmin": g.baseline_tmin,
            "baseline_tmax": g.baseline_tmax,
        }

    result: dict[str, Any] = {"conditions": [cond_to_dict(c) for c in gc.conditions]}
    if gc.groups is not None:
        result["groups"] = [group_to_dict(g) for g in gc.groups]
    return result


def global_conditions_from_dict(data: dict[str, Any]) -> GlobalConditions:
    def cond_from_dict(d: dict[str, Any]) -> ConditionConfig:
        occ = d.get("included_occurrences")
        return ConditionConfig(
            name=d["name"],
            original_name=d["original_name"],
            included_occurrences=tuple(occ) if occ is not None else None,
            duration=d.get("duration", 1.0),
            tmin=d.get("tmin", -2.0),
            tmax=d.get("tmax", 18.0),
            baseline_tmin=d.get("baseline_tmin", -2.0),
            baseline_tmax=d.get("baseline_tmax", 0.0),
        )

    def group_from_dict(d: dict[str, Any]) -> ConditionGroup:
        return ConditionGroup(
            label=d["label"],
            conditions=tuple(d["conditions"]),
            tmin=d.get("tmin", -2.0),
            tmax=d.get("tmax", 18.0),
            baseline_tmin=d.get("baseline_tmin", -2.0),
            baseline_tmax=d.get("baseline_tmax", 0.0),
        )

    conditions = tuple(cond_from_dict(c) for c in data["conditions"])
    groups_data = data.get("groups")
    groups = (
        tuple(group_from_dict(g) for g in groups_data)
        if groups_data
        else None
    )
    return GlobalConditions(conditions=conditions, groups=groups)
