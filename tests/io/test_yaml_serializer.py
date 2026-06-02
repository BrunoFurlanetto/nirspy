"""Tests for yaml_serializer — dump_pipeline overwrite behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from nirspy.domain.data_types import DataType
from nirspy.domain.pipeline import Pipeline
from nirspy.io.yaml_serializer import dump_pipeline
from tests.conftest import make_block

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_pipeline() -> Pipeline:
    """Create a minimal pipeline for serialisation tests."""
    block = make_block("test_block", DataType.RAW, DataType.RAW)
    return Pipeline(name="test-pipe", description="test", steps=[block])


# ---------------------------------------------------------------------------
# Tests — dump_pipeline overwrite guard (S-03)
# ---------------------------------------------------------------------------


class TestDumpPipelineOverwrite:
    """Tests for the overwrite parameter added in S-03."""

    def test_dump_pipeline_writes_when_file_absent(self, tmp_path: Path) -> None:
        """Default behaviour: writing to a new path succeeds."""
        pipeline = _make_simple_pipeline()
        target = tmp_path / "new_pipeline.yml"

        dump_pipeline(pipeline, target)

        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "test-pipe" in content

    def test_dump_pipeline_raises_when_file_exists_and_overwrite_false(
        self, tmp_path: Path
    ) -> None:
        """FileExistsError raised when path exists and overwrite=False (default)."""
        pipeline = _make_simple_pipeline()
        target = tmp_path / "existing.yml"
        target.write_text("pre-existing content", encoding="utf-8")

        with pytest.raises(FileExistsError, match="already exists"):
            dump_pipeline(pipeline, target)

        # File must remain unchanged
        assert target.read_text(encoding="utf-8") == "pre-existing content"

    def test_dump_pipeline_overwrites_when_overwrite_true(
        self, tmp_path: Path
    ) -> None:
        """File is replaced when overwrite=True."""
        pipeline = _make_simple_pipeline()
        target = tmp_path / "overwrite_me.yml"
        target.write_text("old content", encoding="utf-8")

        dump_pipeline(pipeline, target, overwrite=True)

        content = target.read_text(encoding="utf-8")
        assert "old content" not in content
        assert "test-pipe" in content


class TestYamlSerializerNoneDataType:
    """Round-trip with DataType.NONE (T-009)."""

    def test_dump_pipeline_with_none_input_type(self, tmp_path: Path) -> None:
        """A source block with input_type=NONE serializes correctly."""
        source = make_block("load_snirf", DataType.NONE, DataType.RAW)
        consumer = make_block("od", DataType.RAW, DataType.RAW_OD)
        pipeline = Pipeline(
            name="test-none", description="T-009", steps=[source, consumer]
        )
        target = tmp_path / "none_test.yml"
        dump_pipeline(pipeline, target)

        content = target.read_text(encoding="utf-8")
        assert "load_snirf" in content
        assert "schema_version" in content

    def test_round_trip_preserves_none_block(self, tmp_path: Path) -> None:
        """Dump then load pipeline with NONE input_type block."""
        import yaml

        source = make_block("load_snirf", DataType.NONE, DataType.RAW)
        pipeline = Pipeline(
            name="none-rt", description="", steps=[source]
        )
        target = tmp_path / "none_rt.yml"
        dump_pipeline(pipeline, target)

        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert raw["steps"][0]["block_id"] == "load_snirf"


# ---------------------------------------------------------------------------
# GlobalConditions round-trip (T-042)
# ---------------------------------------------------------------------------


class TestYamlRoundTripGlobalConditions:
    """Tests for global_conditions serialisation via dump/load (T-042)."""

    def _make_gc_pipeline(self) -> Pipeline:
        from nirspy.blocks.analysis import BlockAverageBlock, BlockAverageParams
        from nirspy.domain.conditions import ConditionConfig, GlobalConditions

        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    duration=20.0,
                    tmin=-2.0,
                    tmax=22.0,
                    baseline_tmin=-2.0,
                    baseline_tmax=0.0,
                ),
            )
        )
        block = BlockAverageBlock(params=BlockAverageParams())
        return Pipeline(
            name="gc-pipe",
            description="",
            steps=[block],
            global_conditions=gc,
        )

    def _make_gc_pipeline_with_groups(self) -> Pipeline:
        from nirspy.blocks.analysis import BlockAverageBlock, BlockAverageParams
        from nirspy.domain.conditions import (
            ConditionConfig,
            ConditionGroup,
            GlobalConditions,
        )

        gc = GlobalConditions(
            conditions=(
                ConditionConfig(name="TaskA", original_name="1.0", duration=10.0),
                ConditionConfig(name="TaskB", original_name="2.0", duration=12.0),
            ),
            groups=(
                ConditionGroup(
                    label="AllTasks",
                    conditions=("TaskA", "TaskB"),
                    tmin=-2.0,
                    tmax=20.0,
                    baseline_tmin=-2.0,
                    baseline_tmax=0.0,
                ),
            ),
        )
        block = BlockAverageBlock(params=BlockAverageParams())
        return Pipeline(
            name="gc-groups-pipe", description="", steps=[block], global_conditions=gc
        )

    def test_yaml_roundtrip_with_global_conditions(self, tmp_path: Path) -> None:
        """dump_pipeline → load_pipeline preserves global_conditions field-by-field."""
        from nirspy.blocks import registry
        from nirspy.io.yaml_serializer import load_pipeline

        pipeline = self._make_gc_pipeline()
        target = tmp_path / "gc.yml"
        dump_pipeline(pipeline, target)

        loaded = load_pipeline(target, registry)

        assert loaded.global_conditions is not None
        assert len(loaded.global_conditions.conditions) == 1
        cog = loaded.global_conditions.conditions[0]
        assert cog.name == "Cognitive"
        assert cog.original_name == "1.0"
        assert cog.duration == 20.0
        assert cog.tmin == -2.0
        assert cog.tmax == 22.0

    def test_yaml_roundtrip_global_with_groups(self, tmp_path: Path) -> None:
        """dump → load preserves groups inside global_conditions."""
        from nirspy.blocks import registry
        from nirspy.io.yaml_serializer import load_pipeline

        pipeline = self._make_gc_pipeline_with_groups()
        target = tmp_path / "gc_groups.yml"
        dump_pipeline(pipeline, target)

        loaded = load_pipeline(target, registry)

        assert loaded.global_conditions is not None
        assert loaded.global_conditions.groups is not None
        assert len(loaded.global_conditions.groups) == 1
        grp = loaded.global_conditions.groups[0]
        assert grp.label == "AllTasks"
        assert "TaskA" in grp.conditions
        assert "TaskB" in grp.conditions

    def test_yaml_legacy_without_global_conditions(self, tmp_path: Path) -> None:
        """YAML without global_conditions key loads correctly; field is None."""
        import yaml

        from nirspy.blocks import registry
        from nirspy.io.yaml_serializer import load_pipeline

        pipeline_dict = {
            "schema_version": "0.1",
            "name": "legacy",
            "description": "",
            "steps": [],
            "params": {},
            # no global_conditions key
        }
        target = tmp_path / "legacy.yml"
        target.write_text(
            yaml.dump(pipeline_dict, default_flow_style=False), encoding="utf-8"
        )

        loaded = load_pipeline(target, registry)
        assert loaded.global_conditions is None

    def test_yaml_included_occurrences_none(self, tmp_path: Path) -> None:
        """included_occurrences: null in YAML → None in Python."""
        import yaml

        from nirspy.domain.conditions import global_conditions_from_dict

        data: dict[str, object] = {
            "conditions": [
                {
                    "name": "Cog",
                    "original_name": "1.0",
                    "included_occurrences": None,
                    "duration": 10.0,
                    "tmin": -2.0,
                    "tmax": 15.0,
                    "baseline_tmin": -2.0,
                    "baseline_tmax": 0.0,
                }
            ]
        }
        # Simulate YAML round-trip
        yaml_text = yaml.dump(data, default_flow_style=False)
        loaded_data = yaml.safe_load(yaml_text)
        gc = global_conditions_from_dict(loaded_data)

        assert gc.conditions[0].included_occurrences is None

    def test_yaml_included_occurrences_list(self, tmp_path: Path) -> None:
        """included_occurrences list in YAML → tuple in Python."""
        import yaml

        from nirspy.domain.conditions import global_conditions_from_dict

        data: dict[str, object] = {
            "conditions": [
                {
                    "name": "Cog",
                    "original_name": "1.0",
                    "included_occurrences": [0, 2, 4],
                    "duration": 10.0,
                    "tmin": -2.0,
                    "tmax": 15.0,
                    "baseline_tmin": -2.0,
                    "baseline_tmax": 0.0,
                }
            ]
        }
        yaml_text = yaml.dump(data, default_flow_style=False)
        loaded_data = yaml.safe_load(yaml_text)
        gc = global_conditions_from_dict(loaded_data)

        assert isinstance(gc.conditions[0].included_occurrences, tuple)
        assert gc.conditions[0].included_occurrences == (0, 2, 4)

    def test_yaml_roundtrip_idempotent_with_global_conditions(
        self, tmp_path: Path
    ) -> None:
        """dump → load → dump produces identical YAML bytes when gc is present."""
        from nirspy.blocks import registry
        from nirspy.io.yaml_serializer import load_pipeline

        pipeline = self._make_gc_pipeline()
        p1 = tmp_path / "pass1.yml"
        dump_pipeline(pipeline, p1)

        loaded = load_pipeline(p1, registry)
        p2 = tmp_path / "pass2.yml"
        dump_pipeline(loaded, p2)

        assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


class TestYamlRoundTripPerConditionWindows:
    """Round-trip with per_condition_windows (T-012)."""

    def test_round_trip_empty_per_condition(self, tmp_path: Path) -> None:
        """Pipeline with empty per_condition_windows round-trips."""

        from nirspy.blocks import registry
        from nirspy.blocks.analysis import BlockAverageBlock, BlockAverageParams
        from nirspy.io.yaml_serializer import load_pipeline

        params = BlockAverageParams()
        block = BlockAverageBlock(params=params)
        pipeline = Pipeline(
            name="rt-empty", description="", steps=[block]
        )
        target = tmp_path / "empty_pcw.yml"
        dump_pipeline(pipeline, target)

        loaded = load_pipeline(target, registry)
        loaded_params = loaded.steps[0].params
        assert loaded_params.per_condition_windows == {}

    def test_round_trip_with_per_condition(self, tmp_path: Path) -> None:
        """Pipeline with per_condition_windows round-trips correctly."""
        import yaml

        from nirspy.blocks import registry
        from nirspy.blocks.analysis import (
            BlockAverageBlock,
            BlockAverageParams,
            ConditionWindow,
        )
        from nirspy.io.yaml_serializer import load_pipeline

        params = BlockAverageParams(
            per_condition_windows={
                "Tapping": ConditionWindow(
                    tmin=-5.0, tmax=30.0,
                    baseline_tmin=-5.0, baseline_tmax=0.0,
                ),
                "Rest": ConditionWindow(
                    tmin=-2.0, tmax=18.0,
                    baseline_tmin=-2.0, baseline_tmax=0.0,
                ),
            },
        )
        block = BlockAverageBlock(params=params)
        pipeline = Pipeline(
            name="rt-pcw", description="", steps=[block]
        )
        target = tmp_path / "pcw.yml"
        dump_pipeline(pipeline, target)

        # Verify YAML structure
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
        pcw_raw = raw["params"]["block_average"]["per_condition_windows"]
        assert "Tapping" in pcw_raw
        assert pcw_raw["Tapping"]["tmin"] == -5.0

        # Verify reload
        loaded = load_pipeline(target, registry)
        lp = loaded.steps[0].params
        assert isinstance(lp.per_condition_windows["Tapping"], ConditionWindow)
        assert lp.per_condition_windows["Tapping"].tmin == -5.0
        assert lp.per_condition_windows["Rest"].tmax == 18.0

    def test_round_trip_idempotent(self, tmp_path: Path) -> None:
        """Dump -> load -> dump produces identical YAML."""
        from nirspy.blocks import registry
        from nirspy.blocks.analysis import (
            BlockAverageBlock,
            BlockAverageParams,
            ConditionWindow,
        )
        from nirspy.io.yaml_serializer import load_pipeline

        params = BlockAverageParams(
            per_condition_windows={
                "A": ConditionWindow(
                    tmin=-10.0, tmax=60.0,
                    baseline_tmin=-10.0, baseline_tmax=0.0,
                ),
            },
        )
        block = BlockAverageBlock(params=params)
        pipeline = Pipeline(
            name="idempotent", description="", steps=[block]
        )
        p1 = tmp_path / "pass1.yml"
        dump_pipeline(pipeline, p1)

        loaded = load_pipeline(p1, registry)
        p2 = tmp_path / "pass2.yml"
        dump_pipeline(loaded, p2)

        assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
