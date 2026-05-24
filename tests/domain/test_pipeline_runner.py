"""Tests for PipelineRunner (T-023).

Step-by-step executor with params_override support.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pytest

from nirspy.domain.block import Block, BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.execution import ExecutionContext, PipelineRunner
from nirspy.domain.pipeline import Pipeline
from tests.conftest import make_block


@dataclass
class _DummyParams:
    """Params dataclass for testing override."""
    tmin: float = -2.0
    tmax: float = 18.0
    label: str = 'default'


@dataclasses.dataclass
class _ParamsBlock:
    """Block with a params dataclass for override testing."""
    _spec: BlockSpec = dataclasses.field(
        default_factory=lambda: BlockSpec(
            block_id="params_block",
            display_name="ParamsBlock",
            input_type=DataType.RAW,
            output_type=DataType.RAW,
            params_class=_DummyParams,
        )
    )
    params: _DummyParams = dataclasses.field(default_factory=_DummyParams)

    @property
    def spec(self) -> BlockSpec:
        return self._spec

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        return BlockResult(
            data={"tmin": self.params.tmin, "tmax": self.params.tmax, "label": self.params.label},
            block_id="params_block",
        )


def _pipeline(*blocks: Block, name: str = 'test') -> Pipeline:
    return Pipeline(name=name, steps=list(blocks))


class TestPipelineRunnerStepByStep:
    """Core step-by-step behavior."""

    def test_step_through_three_blocks(self) -> None:
        blocks = [
            make_block("a", DataType.NONE, DataType.RAW),
            make_block("b", DataType.RAW, DataType.RAW_OD),
            make_block("c", DataType.RAW_OD, DataType.RAW_HAEMO),
        ]
        runner = PipelineRunner(_pipeline(*blocks))
        runner.start()
        specs = []
        while not runner.is_complete:
            spec = runner.next_block()
            if spec is None:
                break
            specs.append(spec.block_id)
            runner.execute_current()
        assert specs == ["a", "b", "c"]
        assert len(runner.results) == 3
        assert runner.is_complete

    def test_empty_pipeline(self) -> None:
        runner = PipelineRunner(_pipeline())
        runner.start()
        assert runner.is_complete
        assert runner.next_block() is None
        assert runner.results == []

    def test_next_block_before_start_raises(self) -> None:
        runner = PipelineRunner(_pipeline(make_block('x', DataType.NONE, DataType.RAW)))
        with pytest.raises(RuntimeError, match='start'):
            runner.next_block()

    def test_execute_before_next_block_raises(self) -> None:
        runner = PipelineRunner(_pipeline(make_block('x', DataType.NONE, DataType.RAW)))
        runner.start()
        with pytest.raises(RuntimeError, match='next_block'):
            runner.execute_current()

    def test_double_execute_raises(self) -> None:
        runner = PipelineRunner(_pipeline(make_block('x', DataType.NONE, DataType.RAW)))
        runner.start()
        runner.next_block()
        runner.execute_current()
        with pytest.raises(RuntimeError):
            runner.execute_current()

    def test_next_block_after_complete_returns_none(self) -> None:
        runner = PipelineRunner(_pipeline(make_block('x', DataType.NONE, DataType.RAW)))
        runner.start()
        runner.next_block()
        runner.execute_current()
        assert runner.next_block() is None

    def test_current_idx_tracks_position(self) -> None:
        blocks = [
            make_block("a", DataType.NONE, DataType.RAW),
            make_block("b", DataType.RAW, DataType.RAW_OD),
        ]
        runner = PipelineRunner(_pipeline(*blocks))
        runner.start()
        assert runner.current_idx == -1
        runner.next_block()
        assert runner.current_idx == 0
        runner.execute_current()
        runner.next_block()
        assert runner.current_idx == 1

    def test_disabled_blocks_skipped(self) -> None:
        blocks = [
            make_block("a", DataType.NONE, DataType.RAW),
            make_block("b", DataType.RAW, DataType.RAW_OD, enabled=False),
            make_block("c", DataType.RAW, DataType.RAW_HAEMO),
        ]
        runner = PipelineRunner(_pipeline(*blocks))
        runner.start()
        assert runner.total_steps == 2
        ids = []
        while not runner.is_complete:
            spec = runner.next_block()
            if spec is None:
                break
            ids.append(spec.block_id)
            runner.execute_current()
        assert ids == ["a", "c"]


class TestPipelineRunnerParamsOverride:
    """params_override is transient and validated."""

    def test_override_applies_transiently(self) -> None:
        block = _ParamsBlock()  # type: ignore[assignment]
        runner = PipelineRunner(_pipeline(block))
        runner.start()
        runner.next_block()
        result = runner.execute_current(params_override={"tmin": -5.0, "label": "custom"})
        assert result.data["tmin"] == -5.0
        assert result.data["label"] == "custom"
        # Original block params unchanged
        assert block.params.tmin == -2.0
        assert block.params.label == 'default'

    def test_override_unknown_field_raises(self) -> None:
        block = _ParamsBlock()  # type: ignore[assignment]
        runner = PipelineRunner(_pipeline(block))
        runner.start()
        runner.next_block()
        with pytest.raises(ValidationError, match='unknown'):
            runner.execute_current(params_override={"nonexistent": 42})

    def test_override_none_is_noop(self) -> None:
        block = _ParamsBlock()  # type: ignore[assignment]
        runner = PipelineRunner(_pipeline(block))
        runner.start()
        runner.next_block()
        result = runner.execute_current(params_override=None)
        assert result.data["tmin"] == -2.0

    def test_override_empty_dict_is_noop(self) -> None:
        block = _ParamsBlock()  # type: ignore[assignment]
        runner = PipelineRunner(_pipeline(block))
        runner.start()
        runner.next_block()
        result = runner.execute_current(params_override={})
        assert result.data["tmin"] == -2.0


class TestPipelineRunnerMetadata:
    """Metadata propagation via context.extra (ADR-014)."""

    def test_metadata_propagated_to_context(self) -> None:
        block = make_block("a", DataType.NONE, DataType.RAW)
        ctx = ExecutionContext()
        runner = PipelineRunner(_pipeline(block), ctx)
        runner.start()
        runner.next_block()
        runner.execute_current()
        assert "prev_metadata" in ctx.extra

    def test_progress_callback_invoked(self) -> None:
        calls: list[tuple[str, int, int]] = []

        def _progress(block_id: str, step: int, total: int) -> None:
            calls.append((block_id, step, total))

        blocks = [
            make_block("a", DataType.NONE, DataType.RAW),
            make_block("b", DataType.RAW, DataType.RAW_OD),
        ]
        ctx = ExecutionContext(progress=_progress)
        runner = PipelineRunner(_pipeline(*blocks), ctx)
        runner.start()
        while not runner.is_complete:
            spec = runner.next_block()
            if spec is None:
                break
            runner.execute_current()
        assert calls == [("a", 1, 2), ("b", 2, 2)]
