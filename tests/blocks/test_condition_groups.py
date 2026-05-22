"""Tests for ConditionGroup domain + engine (T-024)."""

from __future__ import annotations

from typing import Any

import pytest

from nirspy.blocks.analysis import (
    BlockAverageParams,
    ConditionGroup,
    ConditionWindow,
)
from nirspy.domain.exceptions import ValidationError


class TestConditionGroup:
    """ConditionGroup dataclass basics."""

    def test_creation(self) -> None:
        grp = ConditionGroup(
            label="short",
            condition_names=["S1", "S2"],
            tmin=-2.0, tmax=10.0,
            baseline_tmin=-2.0, baseline_tmax=0.0,
        )
        assert grp.label == "short"
        assert grp.condition_names == ["S1", "S2"]
        assert grp.tmin == -2.0

    def test_frozen(self) -> None:
        grp = ConditionGroup(
            label="x", condition_names=["S1"],
            tmin=-1.0, tmax=5.0,
            baseline_tmin=-1.0, baseline_tmax=0.0,
        )
        with pytest.raises(AttributeError):
            grp.label = "y"  # type: ignore[misc]


class TestBlockAverageParamsMutualExclusion:
    """D3: per_condition_windows and per_condition_groups are mutually exclusive."""

    def test_both_empty_ok(self) -> None:
        params = BlockAverageParams()
        assert params.per_condition_windows == {}
        assert params.per_condition_groups == {}

    def test_only_windows_ok(self) -> None:
        params = BlockAverageParams(
            per_condition_windows={"S1": ConditionWindow(-2, 18, -2, 0)}
        )
        assert len(params.per_condition_windows) == 1

    def test_only_groups_ok(self) -> None:
        params = BlockAverageParams(
            per_condition_groups={
                "short": ConditionGroup(
                    label="short", condition_names=["S1"],
                    tmin=-2, tmax=10, baseline_tmin=-2, baseline_tmax=0,
                )
            }
        )
        assert len(params.per_condition_groups) == 1

    def test_both_non_empty_raises(self) -> None:
        with pytest.raises(ValidationError, match='mutually exclusive'):
            BlockAverageParams(
                per_condition_windows={
                    "S1": ConditionWindow(-2, 18, -2, 0)
                },
                per_condition_groups={
                    "short": ConditionGroup(
                        label="short", condition_names=["S1"],
                        tmin=-2, tmax=10, baseline_tmin=-2, baseline_tmax=0,
                    )
                },
            )


class TestConditionGroupCoercion:
    """Raw dicts coerced to ConditionGroup in __post_init__."""

    def test_dict_coerced_to_group(self) -> None:
        params = BlockAverageParams(
            per_condition_groups={
                "short": {  # type: ignore[dict-item]
                    "label": "short",
                    "condition_names": ["S1", "S2"],
                    "tmin": -2.0, "tmax": 10.0,
                    "baseline_tmin": -2.0, "baseline_tmax": 0.0,
                }
            }
        )
        grp = params.per_condition_groups["short"]
        assert isinstance(grp, ConditionGroup)
        assert grp.label == "short"
        assert grp.condition_names == ["S1", "S2"]


class TestConditionGroupYamlRoundTrip:
    """YAML serialization round-trip for per_condition_groups."""

    def test_round_trip(self, tmp_path: Any) -> None:
        from pathlib import Path

        from nirspy.blocks import registry
        from nirspy.blocks.analysis import BlockAverageBlock
        from nirspy.blocks.load import LoadSnirfBlock, LoadSnirfParams
        from nirspy.domain.pipeline import Pipeline
        from nirspy.io.yaml_serializer import dump_pipeline, load_pipeline

        params = BlockAverageParams(
            per_condition_groups={
                "short": ConditionGroup(
                    label="short", condition_names=["S1", "S2"],
                    tmin=-2.0, tmax=10.0,
                    baseline_tmin=-2.0, baseline_tmax=0.0,
                ),
                "long": ConditionGroup(
                    label="long", condition_names=["S3"],
                    tmin=-5.0, tmax=30.0,
                    baseline_tmin=-5.0, baseline_tmax=0.0,
                ),
            }
        )
        pipeline = Pipeline(
            name="test-groups",
            steps=[
                LoadSnirfBlock(LoadSnirfParams(path="/tmp/test.snirf")),
                BlockAverageBlock(params),
            ],
        )

        path = Path(tmp_path) / 'test_groups.yml'
        dump_pipeline(pipeline, path)
        loaded = load_pipeline(path, registry)

        ba_block = loaded.steps[1]
        assert hasattr(ba_block, 'params')
        loaded_params = ba_block.params
        assert len(loaded_params.per_condition_groups) == 2
        assert "short" in loaded_params.per_condition_groups
        assert "long" in loaded_params.per_condition_groups
        short = loaded_params.per_condition_groups["short"]
        assert isinstance(short, ConditionGroup)
        assert short.condition_names == ["S1", "S2"]
        assert short.tmin == -2.0

    def test_round_trip_idempotent(self, tmp_path: Any) -> None:
        from pathlib import Path

        from nirspy.blocks import registry
        from nirspy.blocks.analysis import BlockAverageBlock
        from nirspy.blocks.load import LoadSnirfBlock, LoadSnirfParams
        from nirspy.domain.pipeline import Pipeline
        from nirspy.io.yaml_serializer import dump_pipeline, load_pipeline

        params = BlockAverageParams(
            per_condition_groups={
                "grp1": ConditionGroup(
                    label="grp1", condition_names=["A"],
                    tmin=-1.0, tmax=5.0,
                    baseline_tmin=-1.0, baseline_tmax=0.0,
                )
            }
        )
        pipeline = Pipeline(
            name="idem",
            steps=[
                LoadSnirfBlock(LoadSnirfParams(path="/tmp/x.snirf")),
                BlockAverageBlock(params),
            ],
        )

        p1 = Path(tmp_path) / 'a.yml'
        p2 = Path(tmp_path) / 'b.yml'
        dump_pipeline(pipeline, p1)
        loaded = load_pipeline(p1, registry)
        dump_pipeline(loaded, p2)
        assert p1.read_text() == p2.read_text()
