"""ExecutionContext, PipelineRunner and run_pipeline_sync."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nirspy.domain.cache import CacheProtocol
    from nirspy.domain.pipeline import Pipeline

from nirspy.domain.block import Block, BlockResult, BlockSpec
from nirspy.domain.exceptions import ExecutionError, ValidationError

# Signature: (block_id, step_index, total_steps) -> None
ProgressCallback = Callable[[str, int, int], None]


def _noop_progress(block_id: str, step: int, total: int) -> None:  # noqa: ARG001
    pass


@dataclass
class ExecutionContext:
    """Carries runtime dependencies injected into each block's ``run`` call.

    All fields are optional — blocks must handle ``None`` gracefully when
    optional resources are absent.
    """

    cache: CacheProtocol | None = None
    """Optional cache adapter; ``None`` means caching is disabled for this run."""

    progress: ProgressCallback = field(default=_noop_progress)
    """Callback invoked after each block completes. Defaults to a no-op."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Extension point for future runners (e.g. distributed context IDs).

    The linear runner stores the previous block's metadata here under
    ``"prev_metadata"`` so downstream blocks can access upstream diagnostics
    (e.g. SCI values for channel pruning). Individual metadata keys are also
    promoted to top-level extra keys when present (e.g. ``"sci_values"``).
    """



class PipelineRunner:
    """Step-by-step pipeline executor.

    Allows the GUI to advance block-by-block, optionally injecting
    ``params_override`` dicts before each execution step. The override
    is **transient** --- it does not mutate the original block's params.

    Usage (interactive)::

        runner = PipelineRunner(pipeline, context)
        runner.start()
        while not runner.is_complete:
            spec = runner.next_block()
            if spec is None:
                break
            result = runner.execute_current(params_override={"tmin": -5.0})

    Usage (headless) --- equivalent to the old ``run_pipeline_sync``::

        runner = PipelineRunner(pipeline, context)
        runner.start()
        while not runner.is_complete:
            runner.next_block()
            runner.execute_current()
    """

    def __init__(
        self,
        pipeline: Pipeline,
        context: ExecutionContext | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._context = context or ExecutionContext()
        self._enabled_steps: list[Block] = []
        self._current_idx: int = -1
        self._results: list[BlockResult] = []
        self._prev_result: BlockResult | None = None
        self._started: bool = False
        self._block_ready: bool = False

    def start(self) -> None:
        """Initialize the runner. Must be called before ``next_block``."""
        self._enabled_steps = [
            step for step in self._pipeline.steps if step.spec.enabled
        ]
        self._current_idx = -1
        self._results = []
        self._prev_result = None
        self._started = True
        self._block_ready = False
        if self._pipeline.global_conditions is not None:
            self._context.extra["global_conditions"] = self._pipeline.global_conditions

    def next_block(self) -> BlockSpec | None:
        """Advance to the next block and return its spec.

        Returns ``None`` when all blocks have been executed (pipeline
        complete).

        Raises
        ------
        RuntimeError
            If ``start()`` was not called first.
        """
        if not self._started:
            raise RuntimeError(
                "PipelineRunner.start() must be called before next_block()."
            )
        next_idx = self._current_idx + 1
        if next_idx >= len(self._enabled_steps):
            self._block_ready = False
            return None
        self._current_idx = next_idx
        self._block_ready = True
        return self._enabled_steps[self._current_idx].spec

    def execute_current(
        self,
        params_override: dict[str, Any] | None = None,
    ) -> BlockResult:
        """Execute the current block, optionally with transient param overrides.

        Parameters
        ----------
        params_override:
            Dict of param field names to override values. Applied
            transiently --- the original block's ``params`` attribute is
            **not** mutated.

        Raises
        ------
        RuntimeError
            If ``next_block()`` was not called or the pipeline is complete.
        ValidationError
            If ``params_override`` contains unknown field names.
        """
        if not self._block_ready:
            raise RuntimeError(
                "PipelineRunner.execute_current() called without a "
                "preceding next_block() call, or the pipeline is complete."
            )

        block = self._enabled_steps[self._current_idx]
        total = len(self._enabled_steps)

        # Build inputs dict (same logic as old run_pipeline_sync)
        if self._prev_result is None:
            inputs: dict[str, Any] = {}
        else:
            inputs = {self._prev_result.block_id: self._prev_result.data}

        # Apply transient params override if provided
        if params_override and hasattr(block, "params"):
            block = self._apply_params_override(block, params_override)

        try:
            result = block.run(self._context, inputs)
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(
                f"Block '{block.spec.block_id}' failed at step "
                f"{self._current_idx + 1}/{total}: {exc}"
            ) from exc

        # Propagate metadata to context.extra for downstream blocks (ADR-014)
        self._context.extra["prev_metadata"] = result.metadata
        for key, value in result.metadata.items():
            self._context.extra[key] = value

        self._context.progress(
            block.spec.block_id, self._current_idx + 1, total,
        )
        self._results.append(result)
        self._prev_result = result
        self._block_ready = False

        return result

    @property
    def is_complete(self) -> bool:
        """Whether all enabled blocks have been executed."""
        if not self._started:
            return False
        return (
            self._current_idx >= len(self._enabled_steps) - 1
            and not self._block_ready
        )

    @property
    def current_idx(self) -> int:
        """Index of the current block (0-based among enabled steps)."""
        return self._current_idx

    @property
    def results(self) -> list[BlockResult]:
        """List of results from executed blocks so far."""
        return list(self._results)

    @property
    def total_steps(self) -> int:
        """Total number of enabled steps."""
        return len(self._enabled_steps)

    @property
    def current_block(self) -> Block | None:
        """The current block instance, or None if not started/complete."""
        if not self._started or self._current_idx < 0:
            return None
        if self._current_idx >= len(self._enabled_steps):
            return None
        return self._enabled_steps[self._current_idx]

    @staticmethod
    def _apply_params_override(
        block: Block,
        overrides: dict[str, Any],
    ) -> Block:
        """Create a copy of block with overridden params (transient).

        The original block is never mutated. A new block instance is
        created with the merged params dataclass.

        Raises
        ------
        ValidationError
            If any key in overrides is not a valid field of the params
            dataclass.
        """
        params = getattr(block, "params", None)
        if params is None or not dataclasses.is_dataclass(params):
            raise ValidationError(
                f"Block '{block.spec.block_id}' has no dataclass params "
                f"--- cannot apply params_override."
            )

        # Validate override keys
        valid_fields = {f.name for f in dataclasses.fields(params)}
        unknown = set(overrides) - valid_fields
        if unknown:
            raise ValidationError(
                f"params_override contains unknown field(s) for "
                f"'{block.spec.block_id}': {sorted(unknown)}. "
                f"Valid fields: {sorted(valid_fields)}."
            )

        # Merge: original params + overrides
        # cast needed because is_dataclass() does not narrow for mypy
        params_dict: dict[str, Any] = dataclasses.asdict(params)  # type: ignore[arg-type]
        merged = {**params_dict, **overrides}
        params_cls: type[Any] = type(params)
        new_params = params_cls(**merged)

        # Create new block instance with merged params
        block_cls = type(block)
        return block_cls(params=new_params)  # type: ignore[call-arg]


def run_pipeline_sync(
    pipeline: Pipeline,
    context: ExecutionContext | None = None,
) -> list[BlockResult]:
    """Execute pipeline synchronously --- backward-compatible wrapper.

    This is a thin wrapper around :class:`PipelineRunner` that calls
    ``next_block()`` + ``execute_current()`` in a loop until complete.
    Behavior is identical to the pre-T-023 implementation.

    Parameters
    ----------
    pipeline:
        The :class:`~nirspy.domain.pipeline.Pipeline` to execute.
    context:
        Optional :class:`ExecutionContext`. A default (no cache, no-op progress)
        is created when omitted.

    Returns
    -------
    list[BlockResult]
        One result per enabled block, in execution order.

    Raises
    ------
    ExecutionError
        Wraps any exception raised by a block's ``run`` method.
    """
    runner = PipelineRunner(pipeline, context)
    runner.start()
    while not runner.is_complete:
        spec = runner.next_block()
        if spec is None:
            break
        runner.execute_current()
    return runner.results
