"""Tests for ExecutionContext and run_pipeline_sync.

ADR-009: Block.run(context, inputs) — params live in self.params.
run_pipeline_sync no longer accepts initial_data.
  - First block  → inputs={}
  - Subsequent   → inputs={prev_block_id: prev_result.data}
"""

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
# Inline block helpers (new ADR-009 signature)
# ---------------------------------------------------------------------------


def _make_broken_block() -> Block:
    """Block whose run() raises ValueError."""

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

        def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
            raise ValueError("intentional error")

    return _BrokenBlock()  # type: ignore[return-value]


def _make_fail_block() -> Block:
    """Block whose run() raises RuntimeError."""

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

        def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
            raise RuntimeError("original cause")

    return _FailBlock()  # type: ignore[return-value]


def _make_disabled_fail_block() -> Block:
    """Disabled block whose run() raises RuntimeError (should never be called)."""

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

        def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
            raise RuntimeError("should not be called")

    return _DisabledFail()  # type: ignore[return-value]


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
        results = run_pipeline_sync(p)
        assert results == []

    def test_single_block_returns_one_result(self) -> None:
        block = make_block("b1", DataType.RAW, DataType.RAW_OD)
        p = _make_pipeline(block)
        results = run_pipeline_sync(p)
        assert len(results) == 1
        assert isinstance(results[0], BlockResult)

    def test_data_passes_through_chain(self) -> None:
        # FakeBlock forwards the upstream value; in a two-block chain the
        # second block receives the first block's output data (None, since the
        # first block has no upstream).
        b1 = make_block("b1", DataType.RAW, DataType.RAW)
        b2 = make_block("b2", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2)
        results = run_pipeline_sync(p)
        # b2 receives {b1.block_id: b1_result.data} — b1 returned None (no
        # upstream) so b2 also returns None.
        assert results[-1].data is None

    def test_results_count_equals_enabled_blocks(self) -> None:
        b1 = make_block("b1", DataType.RAW, DataType.RAW)
        b2 = make_block("b2", DataType.RAW, DataType.RAW, enabled=False)
        b3 = make_block("b3", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2, b3)
        results = run_pipeline_sync(p)
        assert len(results) == 2  # b2 is disabled

    def test_block_ids_in_results_match_enabled_blocks(self) -> None:
        b1 = make_block("first", DataType.RAW, DataType.RAW)
        b2 = make_block("second", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2)
        results = run_pipeline_sync(p)
        assert [r.block_id for r in results] == ["first", "second"]

    def test_none_context_creates_default(self) -> None:
        block = make_block("b1", DataType.RAW, DataType.RAW)
        p = _make_pipeline(block)
        results = run_pipeline_sync(p, context=None)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# inputs contract verification
# ---------------------------------------------------------------------------


class TestRunPipelineSyncInputsContract:
    """Verify ADR-009 inputs dict semantics."""

    def test_first_block_receives_empty_inputs(self) -> None:
        """The executor must call first block with inputs={}."""
        received_inputs: list[dict[str, Any]] = []

        @dataclasses.dataclass
        class _CapturingBlock:
            _spec: BlockSpec = dataclasses.field(
                default_factory=lambda: BlockSpec(
                    block_id="capturer",
                    display_name="C",
                    input_type=DataType.ANY,
                    output_type=DataType.RAW,
                )
            )

            @property
            def spec(self) -> BlockSpec:
                return self._spec

            def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
                received_inputs.append(dict(inputs))
                return BlockResult(data="first_output", block_id="capturer")

        block = _CapturingBlock()
        p = _make_pipeline(block)  # type: ignore[arg-type]
        run_pipeline_sync(p)
        assert received_inputs == [{}]

    def test_second_block_receives_first_block_output(self) -> None:
        """Second block must receive {first_block_id: first_result.data}."""
        received_inputs: list[dict[str, Any]] = []

        @dataclasses.dataclass
        class _ProducerBlock:
            _spec: BlockSpec = dataclasses.field(
                default_factory=lambda: BlockSpec(
                    block_id="producer",
                    display_name="P",
                    input_type=DataType.ANY,
                    output_type=DataType.RAW,
                )
            )

            @property
            def spec(self) -> BlockSpec:
                return self._spec

            def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
                return BlockResult(data="produced_value", block_id="producer")

        @dataclasses.dataclass
        class _ConsumerBlock:
            _spec: BlockSpec = dataclasses.field(
                default_factory=lambda: BlockSpec(
                    block_id="consumer",
                    display_name="C",
                    input_type=DataType.RAW,
                    output_type=DataType.RAW,
                )
            )

            @property
            def spec(self) -> BlockSpec:
                return self._spec

            def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
                received_inputs.append(dict(inputs))
                return BlockResult(data=inputs.get("producer"), block_id="consumer")

        p = _make_pipeline(  # type: ignore[arg-type]
            _ProducerBlock(), _ConsumerBlock()
        )
        run_pipeline_sync(p)
        assert received_inputs == [{"producer": "produced_value"}]

    def test_linear_chain_key_is_previous_block_id(self) -> None:
        """Key in inputs must be the block_id of the previous block."""
        seen_keys: list[list[str]] = []

        @dataclasses.dataclass
        class _RecorderBlock:
            _bid: str

            @property
            def spec(self) -> BlockSpec:
                return BlockSpec(
                    block_id=self._bid,
                    display_name=self._bid,
                    input_type=DataType.RAW,
                    output_type=DataType.RAW,
                )

            def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
                seen_keys.append(list(inputs.keys()))
                return BlockResult(data=f"out_{self._bid}", block_id=self._bid)

        b1 = _RecorderBlock("step1")
        b2 = _RecorderBlock("step2")
        b3 = _RecorderBlock("step3")
        p = _make_pipeline(b1, b2, b3)  # type: ignore[arg-type]
        run_pipeline_sync(p)
        assert seen_keys[0] == []          # first block: empty inputs
        assert seen_keys[1] == ["step1"]   # second block: key = step1
        assert seen_keys[2] == ["step2"]   # third block: key = step2


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
        run_pipeline_sync(p, context=ctx)
        assert len(calls) == 2

    def test_progress_step_indices_are_correct(self) -> None:
        blocks = [make_block(f"b{i}", DataType.RAW, DataType.RAW) for i in range(3)]
        p = _make_pipeline(*blocks)
        calls, cb = _collect_progress()
        ctx = ExecutionContext(progress=cb)
        run_pipeline_sync(p, context=ctx)
        steps = [c[1] for c in calls]
        assert steps == [1, 2, 3]

    def test_progress_total_is_number_of_enabled_blocks(self) -> None:
        b1 = make_block("b1", DataType.RAW, DataType.RAW)
        b2 = make_block("b2", DataType.RAW, DataType.RAW, enabled=False)
        b3 = make_block("b3", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2, b3)
        calls, cb = _collect_progress()
        ctx = ExecutionContext(progress=cb)
        run_pipeline_sync(p, context=ctx)
        totals = {c[2] for c in calls}
        assert totals == {2}

    def test_progress_block_ids_match_enabled_blocks(self) -> None:
        b1 = make_block("alpha", DataType.RAW, DataType.RAW)
        b2 = make_block("beta", DataType.RAW, DataType.RAW)
        p = _make_pipeline(b1, b2)
        calls, cb = _collect_progress()
        ctx = ExecutionContext(progress=cb)
        run_pipeline_sync(p, context=ctx)
        reported_ids = [c[0] for c in calls]
        assert reported_ids == ["alpha", "beta"]

    def test_no_progress_called_for_empty_pipeline(self) -> None:
        p = Pipeline(name="empty")
        calls, cb = _collect_progress()
        ctx = ExecutionContext(progress=cb)
        run_pipeline_sync(p, context=ctx)
        assert calls == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestRunPipelineSyncErrors:
    def test_block_exception_wrapped_in_execution_error(self) -> None:
        block = _make_broken_block()
        p = Pipeline(name="failing", steps=[block])  # type: ignore[arg-type]
        with pytest.raises(ExecutionError, match="broken"):
            run_pipeline_sync(p)

    def test_execution_error_wraps_original_cause(self) -> None:
        block = _make_fail_block()
        p = Pipeline(name="fail", steps=[block])  # type: ignore[arg-type]
        with pytest.raises(ExecutionError) as exc_info:
            run_pipeline_sync(p)
        assert exc_info.value.__cause__ is not None

    def test_disabled_failing_block_does_not_raise(self) -> None:
        block = _make_disabled_fail_block()
        p = Pipeline(name="ok", steps=[block])  # type: ignore[arg-type]
        results = run_pipeline_sync(p)
        assert results == []
