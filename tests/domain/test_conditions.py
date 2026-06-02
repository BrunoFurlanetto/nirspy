"""Unit tests for nirspy.domain.conditions.

Covers ConditionConfig, ConditionGroup, GlobalConditions and resolve_conditions.

The module may be partially implemented while DBA work is in progress.  Any
symbol that is not yet present in the module causes the test class that
depends on it to be skipped gracefully via the module-level skip guards at
the bottom of the import block.
"""

from __future__ import annotations

import logging
import math

import pytest

# The module must exist for any test to run.
conditions_mod = pytest.importorskip(
    "nirspy.domain.conditions",
    reason="nirspy.domain.conditions not yet available",
)

# ── symbol-level guards ────────────────────────────────────────────────────────
# Each symbol is fetched with getattr so that missing names produce a
# module-level skip marker rather than an ImportError that would fail the whole
# collection.

_ConditionConfig = getattr(conditions_mod, "ConditionConfig", None)
_ConditionGroup = getattr(conditions_mod, "ConditionGroup", None)
_GlobalConditions = getattr(conditions_mod, "GlobalConditions", None)
_resolve_conditions = getattr(conditions_mod, "resolve_conditions", None)

_SKIP_CONDITION_CONFIG = _ConditionConfig is None
_SKIP_CONDITION_GROUP = _ConditionGroup is None
_SKIP_GLOBAL_CONDITIONS = _GlobalConditions is None
_SKIP_RESOLVE = _resolve_conditions is None

# Re-export under canonical names for use in tests (only safe when guard is off)
if not _SKIP_CONDITION_CONFIG:
    ConditionConfig = _ConditionConfig  # type: ignore[assignment]
if not _SKIP_CONDITION_GROUP:
    ConditionGroup = _ConditionGroup  # type: ignore[assignment]
if not _SKIP_GLOBAL_CONDITIONS:
    GlobalConditions = _GlobalConditions  # type: ignore[assignment]
if not _SKIP_RESOLVE:
    resolve_conditions = _resolve_conditions  # type: ignore[assignment]


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def cog_cond():  # noqa: ANN201
    return ConditionConfig(name="Cognitive", original_name="1.0", duration=10.0)


@pytest.fixture
def motor_cond():  # noqa: ANN201
    return ConditionConfig(name="Motor", original_name="2.0", duration=5.0)


@pytest.fixture
def two_conditions(cog_cond, motor_cond):  # noqa: ANN201
    return GlobalConditions(conditions=(cog_cond, motor_cond))


@pytest.fixture
def conditions_with_group(cog_cond, motor_cond):  # noqa: ANN201
    return GlobalConditions(
        conditions=(cog_cond, motor_cond),
        groups=(ConditionGroup(label="Task", conditions=("Cognitive", "Motor")),),
    )


# ── ConditionConfig ────────────────────────────────────────────────────────────


@pytest.mark.skipif(_SKIP_CONDITION_CONFIG, reason="ConditionConfig not yet implemented")
class TestConditionConfigValid:
    """ConditionConfig accepts well-formed inputs."""

    def test_minimal_construction_only_required_fields(self) -> None:
        """Only name and original_name are required; defaults must not raise."""
        cfg = ConditionConfig(name="Cognitive", original_name="1.0")
        assert cfg.name == "Cognitive"
        assert cfg.original_name == "1.0"

    def test_default_duration_is_positive_finite(self) -> None:
        cfg = ConditionConfig(name="A", original_name="a")
        assert cfg.duration > 0
        assert math.isfinite(cfg.duration)

    def test_included_occurrences_none_is_valid(self) -> None:
        cfg = ConditionConfig(
            name="A",
            original_name="a",
            included_occurrences=None,
        )
        assert cfg.included_occurrences is None

    def test_included_occurrences_tuple_is_valid(self) -> None:
        cfg = ConditionConfig(
            name="A",
            original_name="a",
            included_occurrences=(0, 1, 3),
        )
        assert cfg.included_occurrences == (0, 1, 3)

    def test_instance_is_immutable(self) -> None:
        """Frozen dataclass — assignment must raise."""
        cfg = ConditionConfig(name="A", original_name="a")
        with pytest.raises((AttributeError, TypeError)):
            cfg.name = "B"  # type: ignore[misc]

    def test_explicit_valid_window(self) -> None:
        """Custom tmin < tmax accepted."""
        cfg = ConditionConfig(name="A", original_name="a", tmin=-5.0, tmax=20.0)
        assert cfg.tmin < cfg.tmax

    def test_explicit_valid_baseline(self) -> None:
        """baseline_tmin <= baseline_tmax accepted (equal is valid)."""
        cfg = ConditionConfig(
            name="A",
            original_name="a",
            baseline_tmin=-2.0,
            baseline_tmax=-2.0,
        )
        assert cfg.baseline_tmin == cfg.baseline_tmax


@pytest.mark.skipif(_SKIP_CONDITION_CONFIG, reason="ConditionConfig not yet implemented")
class TestConditionConfigInvalid:
    """ConditionConfig rejects malformed inputs with ValueError."""

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ConditionConfig(name="", original_name="1.0")

    def test_empty_original_name_raises(self) -> None:
        with pytest.raises(ValueError, match="original_name"):
            ConditionConfig(name="A", original_name="")

    def test_duration_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            ConditionConfig(name="A", original_name="a", duration=0.0)

    def test_duration_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            ConditionConfig(name="A", original_name="a", duration=-1.0)

    def test_duration_infinite_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            ConditionConfig(name="A", original_name="a", duration=float("inf"))

    def test_tmin_equal_tmax_raises(self) -> None:
        with pytest.raises(ValueError, match="tmin"):
            ConditionConfig(name="A", original_name="a", tmin=0.0, tmax=0.0)

    def test_tmin_greater_than_tmax_raises(self) -> None:
        with pytest.raises(ValueError, match="tmin"):
            ConditionConfig(name="A", original_name="a", tmin=5.0, tmax=-2.0)

    def test_baseline_tmin_greater_than_baseline_tmax_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline"):
            ConditionConfig(
                name="A",
                original_name="a",
                baseline_tmin=1.0,
                baseline_tmax=0.0,
            )


# ── ConditionGroup ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(_SKIP_CONDITION_GROUP, reason="ConditionGroup not yet implemented")
class TestConditionGroupValid:
    """ConditionGroup accepts well-formed inputs."""

    def test_minimal_construction(self) -> None:
        grp = ConditionGroup(label="Task", conditions=("Cognitive", "Motor"))
        assert grp.label == "Task"
        assert "Cognitive" in grp.conditions

    def test_conditions_with_single_item(self) -> None:
        grp = ConditionGroup(label="Solo", conditions=("OnlyOne",))
        assert len(grp.conditions) == 1

    def test_instance_is_immutable(self) -> None:
        grp = ConditionGroup(label="G", conditions=("A", "B"))
        with pytest.raises((AttributeError, TypeError)):
            grp.label = "X"  # type: ignore[misc]


@pytest.mark.skipif(_SKIP_CONDITION_GROUP, reason="ConditionGroup not yet implemented")
class TestConditionGroupInvalid:
    """ConditionGroup rejects malformed inputs with ValueError."""

    def test_empty_label_raises(self) -> None:
        with pytest.raises(ValueError, match="label"):
            ConditionGroup(label="", conditions=("A",))

    def test_empty_conditions_raises(self) -> None:
        with pytest.raises(ValueError, match="conditions"):
            ConditionGroup(label="G", conditions=())

    def test_tmin_equal_tmax_raises(self) -> None:
        with pytest.raises(ValueError, match="tmin"):
            ConditionGroup(label="G", conditions=("A",), tmin=0.0, tmax=0.0)

    def test_tmin_greater_than_tmax_raises(self) -> None:
        with pytest.raises(ValueError, match="tmin"):
            ConditionGroup(label="G", conditions=("A",), tmin=5.0, tmax=1.0)


# ── GlobalConditions ───────────────────────────────────────────────────────────


@pytest.mark.skipif(
    _SKIP_CONDITION_CONFIG or _SKIP_GLOBAL_CONDITIONS,
    reason="ConditionConfig or GlobalConditions not yet implemented",
)
class TestGlobalConditionsValid:
    """GlobalConditions accepts well-formed inputs."""

    def test_two_conditions_no_groups(self, two_conditions) -> None:  # noqa: ANN001
        assert len(two_conditions.conditions) == 2

    def test_conditions_with_group(self, conditions_with_group) -> None:  # noqa: ANN001
        assert len(conditions_with_group.groups) == 1

    def test_conditions_attribute_is_tuple(self, two_conditions) -> None:  # noqa: ANN001
        assert isinstance(two_conditions.conditions, tuple)


@pytest.mark.skipif(
    _SKIP_CONDITION_CONFIG or _SKIP_GLOBAL_CONDITIONS,
    reason="ConditionConfig or GlobalConditions not yet implemented",
)
class TestGlobalConditionsInvalid:
    """GlobalConditions rejects malformed inputs with ValueError."""

    def test_empty_conditions_raises(self) -> None:
        with pytest.raises(ValueError, match="conditions"):
            GlobalConditions(conditions=())

    def test_duplicate_condition_names_raise(self, cog_cond) -> None:  # noqa: ANN001
        dup = ConditionConfig(name="Cognitive", original_name="1.0_dup")
        with pytest.raises(ValueError, match="[Dd]uplicate|name"):
            GlobalConditions(conditions=(cog_cond, dup))

    def test_duplicate_group_labels_raise(self, two_conditions) -> None:  # noqa: ANN001
        grp_a = ConditionGroup(label="Task", conditions=("Cognitive",))
        grp_b = ConditionGroup(label="Task", conditions=("Motor",))
        with pytest.raises(ValueError, match="[Dd]uplicate|label"):
            GlobalConditions(
                conditions=two_conditions.conditions,
                groups=(grp_a, grp_b),
            )

    def test_group_references_unknown_condition_raises(
        self, two_conditions  # noqa: ANN001
    ) -> None:
        grp = ConditionGroup(label="Bad", conditions=("DoesNotExist",))
        with pytest.raises(ValueError, match="[Uu]nknown|condition|DoesNotExist"):
            GlobalConditions(
                conditions=two_conditions.conditions,
                groups=(grp,),
            )


# ── resolve_conditions ─────────────────────────────────────────────────────────


@pytest.mark.skipif(
    _SKIP_CONDITION_CONFIG or _SKIP_GLOBAL_CONDITIONS or _SKIP_RESOLVE,
    reason="resolve_conditions or its dependencies not yet implemented",
)
class TestResolveConditions:
    """resolve_conditions behaviour with various context_extra inputs."""

    def test_missing_global_conditions_key_returns_none(self) -> None:
        result = resolve_conditions(context_extra={}, local_condition_params={})
        assert result is None

    def test_no_global_key_with_local_params_returns_none(self) -> None:
        result = resolve_conditions(
            context_extra={"other_key": "value"},
            local_condition_params={"tmin": -2.0},
        )
        assert result is None

    def test_global_conditions_present_returns_resolved(
        self, two_conditions  # noqa: ANN001
    ) -> None:
        result = resolve_conditions(
            context_extra={"global_conditions": two_conditions},
            local_condition_params={},
        )
        assert result is not None
        assert result.source == "global"

    def test_global_plus_local_params_warns(
        self,
        two_conditions,  # noqa: ANN001
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="nirspy.domain.conditions"):
            result = resolve_conditions(
                context_extra={"global_conditions": two_conditions},
                local_condition_params={"tmin": -5.0},
            )
        assert result is not None
        assert result.source == "global"
        assert any(
            record.levelno >= logging.WARNING for record in caplog.records
        ), "Expected at least one WARNING when local_condition_params is non-empty"
