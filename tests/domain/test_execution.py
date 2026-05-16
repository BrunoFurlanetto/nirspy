"""Tests for ExecutionContext and run_pipeline_sync."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from nirspy.domain.block import Block, BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ExecutionError
from nirspy.domain.execution import ExecutionContext, ProgressCallback, run_pipeline_sync
from nirspy.domain.pipeline import Pipeline
from nirspy.engine.cache_adapter import InMemoryCacheAdapter
from tests.conftest import make_block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(*blocks: Block, name: str = "test") -> Pipeline:
    return Pipeline(name=name, steps=list(blocks))


def _collect_progress() -> tuple[list[tuple[str, int, int]], ProgressCallback]:
    calls: list[tuple[str, int, int]] = []

    def cb(block_id: str, step: int, total: int) -> None:
        calls.append((block_id, step, total))

    return calls, cb


# ---------------------------------------------------------------------------
# ExecutionContext
# ---------------------------------------------------------------------------


class TestExecutionContext:
    def test_default_context_has_no_cache(self) -> None:
        ctx = ExecutionContext()
        assert ctx.cache is None

    def test_default_context_progress_is_callable(self) -> None:
        ctx = ExecutionContext()
        # default progress is a no-op callable; calling it must not raise
        ctx.progress("block_id", 1, 1)

    def test_context_with_cache(self) -> None:
        cache = InMemoryCacheAdapter()
        ctx = ExecutionContext(cache=cache)
        assert ctx.cache is cache

    def test_context_with_custom_progress(self) -> None:
        called: list[tuple[str, int, int]] = []

        def cb(block_id: str, step: int, total: int) -> None:
            called.append((block_id, step, total))

        ctx = ExecutionContext(progress=cb)
        ctx.progress("b1", 1, 2)
        assert called == [("b1", 1, 2)]

    def test_extra_dict_is_empty_by_default(self) -> None:
        ctx = ExecutionContext()
        assert ctx.extra == {}

    def test_extra_dict_is_independent_per_instance(self) -> None:
        ctx_a = ExecutionContext()
        ctx_b = ExecutionContext()
        ctx_a.extra["x"] = 1
        assert "x" not in ctx_b.extra


# ---------------------------------------------------------------------------
# run_pipeline_sync — happy path
# ---------------------------------------------------------------------------


class TestRunPipelineSyncHappyPath:
    def test_empty_pipeline_returns_empty_list(self) -> None:
        p = Pipeline(name="empty")
        results = run_pipeline_sync(p, initial_data=None)
        assert results == []

    def test_single_block_returns_one_result(self) -> None:
        block = make_block("b1", DataType.RAW, DataType.RAW_OD)
        p = _make_pipeline(block)
        results = run_pipeline_sync(p, initial_data="data")
        assert len(results) == 1
        assert isinstance(results[0], BlockResult)

    def test_data_passes_through_chain(self) -> None:
        b1 = make_block("b1", DataType.RAW, DataType.RAW)
        b2 = make_block("b2", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2)
        sentinel = object()
        results = run_pipeline_sync(p, initial_data=sentinel)
        # FakeBlock passes data unchanged; last result should carry sentinel
        assert results[-1].data is sentinel

    def test_results_count_equals_enabled_blocks(self) -> None:
        b1 = make_block("b1", DataType.RAW, DataType.RAW)
        b2 = make_block("b2", DataType.RAW, DataType.RAW, enabled=False)
        b3 = make_block("b3", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2, b3)
        results = run_pipeline_sync(p, initial_data="x")
        assert len(results) == 2  # b2 is disabled

    def test_block_ids_in_results_match_enabled_blocks(self) -> None:
        b1 = make_block("first", DataType.RAW, DataType.RAW)
        b2 = make_block("second", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2)
        results = run_pipeline_sync(p, initial_data=0)
        assert [r.block_id for r in results] == ["first", "second"]

    def test_none_context_creates_default(self) -> None:
        block = make_block("b1", DataType.RAW, DataType.RAW)
        p = _make_pipeline(block)
        results = run_pipeline_sync(p, initial_data=1, context=None)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# ProgressCallback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_progress_called_once_per_enabled_block(self) -> None:
        b1 = make_block("b1", DataType.RAW, DataType.RAW)
        b2 = make_block("b2", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2)
        calls, cb = _collect_progress()
        ctx = ExecutionContext(progress=cb)
        run_pipeline_sync(p, initial_data="x", context=ctx)
        assert len(calls) == 2

    def test_progress_step_indices_are_correct(self) -> None:
        blocks = [make_block(f"b{i}", DataType.RAW, DataType.RAW) for i in range(3)]
        p = _make_pipeline(*blocks)
        calls, cb = _collect_progress()
        ctx = ExecutionContext(progress=cb)
        run_pipeline_sync(p, initial_data="x", context=ctx)
        steps = [c[1] for c in calls]
        assert steps == [1, 2, 3]

    def test_progress_total_is_number_of_enabled_blocks(self) -> None:
        b1 = make_block("b1", DataType.RAW, DataType.RAW)
        b2 = make_block("b2", DataType.RAW, DataType.RAW, enabled=False)
        b3 = make_block("b3", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2, b3)
        calls, cb = _collect_progress()
        ctx = ExecutionContext(progress=cb)
        run_pipeline_sync(p, initial_data="x", context=ctx)
        totals = {c[2] for c in calls}
        assert totals == {2}

    def test_progress_block_ids_match_enabled_blocks(self) -> None:
        b1 = make_block("alpha", DataType.RAW, DataType.RAW)
        b2 = make_block("beta", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2)
        calls, cb = _collect_progress()
        ctx = ExecutionContext(progress=cb)
        run_pipeline_sync(p, initial_data=None, context=ctx)
        reported_ids = [c[0] for c in calls]
        assert reported_ids == ["alpha", "beta"]

    def test_no_progress_called_for_empty_pipeline(self) -> None:
        p = Pipeline(name="empty")
        calls, cb = _collect_progress()
        ctx = ExecutionContext(progress=cb)
        run_pipeline_sync(p, initial_data=None, context=ctx)
        assert calls == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestRunPipelineSyncErrors:
    def test_block_exception_wrapped_in_execution_error(self) -> None:
        @dataclasses.dataclass
        class _BrokenBlock:
            _spec: BlockSpec = dataclasses.field(
                default_factory=lambda: BlockSpec(
                    block_id="broken",
                    display_name="Broken",
                    input_type=DataType.RAW,
                    output_type=DataType.RAW_OD,
                )
            )

            @property
            def spec(self) -> BlockSpec:
                return self._spec

            def run(self, data: Any, params: Any, context: Any) -> BlockResult:
                raise ValueError("intentional error")

        block = _BrokenBlock()
        p = Pipeline(name="failing", steps=[block])  # type: ignore[arg-type]
        with pytest.raises(ExecutionError, match="broken"):
            run_pipeline_sync(p, initial_data="data")

    def test_execution_error_wraps_original_cause(self) -> None:
        @dataclasses.dataclass
        class _FailBlock:
            _spec: BlockSpec = dataclasses.field(
                default_factory=lambda: BlockSpec(
                    block_id="fail_block",
                    display_name="Fail",
                    input_type=DataType.RAW,
                    output_type=DataType.RAW_OD,
                )
            )

            @property
            def spec(self) -> BlockSpec:
                return self._spec

            def run(self, data: Any, params: Any, context: Any) -> BlockResult:
                raise RuntimeError("original cause")

        block = _FailBlock()
        p = Pipeline(name="fail", steps=[block])  # type: ignore[arg-type]
        with pytest.raises(ExecutionError) as exc_info:
            run_pipeline_sync(p, initial_data=None)
        assert exc_info.value.__cause__ is not None

    def test_disabled_failing_block_does_not_raise(self) -> None:
        @dataclasses.dataclass
        class _DisabledFail:
            _spec: BlockSpec = dataclasses.field(
                default_factory=lambda: BlockSpec(
                    block_id="dfail",
                    display_name="Disabled Fail",
                    input_type=DataType.RAW,
                    output_type=DataType.RAW_OD,
                    enabled=False,
                )
            )

            @property
            def spec(self) -> BlockSpec:
                return self._spec

            def run(self, data: Any, params: Any, context: Any) -> BlockResult:
                raise RuntimeError("should not be called")

        block = _DisabledFail()
        p = Pipeline(name="ok", steps=[block])  # type: ignore[arg-type]
        results = run_pipeline_sync(p, initial_data="data")
        assert results == []
