"""ExecutionContext, ProgressCallback and run_pipeline_sync."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nirspy.domain.cache import CacheProtocol
    from nirspy.domain.pipeline import Pipeline

from nirspy.domain.block import BlockResult
from nirspy.domain.exceptions import ExecutionError

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


def run_pipeline_sync(
    pipeline: Pipeline,
    context: ExecutionContext | None = None,
) -> list[BlockResult]:
    """Execute *pipeline* synchronously, passing data through enabled blocks in order.

    The first block in the pipeline receives ``inputs={}`` — it is responsible
    for loading or generating its own data (e.g. reading from disk via its own
    ``self.params``).  Each subsequent block receives
    ``inputs={prev_block_id: prev_result.data}``.

    Metadata propagation (ADR-014):
        After each block completes, its ``BlockResult.metadata`` is stored in
        ``context.extra["prev_metadata"]``. Individual keys are also promoted
        to ``context.extra`` for convenient access by downstream blocks
        (e.g. ``context.extra["sci_values"]``).

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
        One result per *enabled* block, in execution order.

    Raises
    ------
    ExecutionError
        Wraps any exception raised by a block's ``run`` method.
    """
    if context is None:
        context = ExecutionContext()

    enabled_steps = [step for step in pipeline.steps if step.spec.enabled]
    total = len(enabled_steps)
    results: list[BlockResult] = []
    prev_result: BlockResult | None = None

    for idx, block in enumerate(enabled_steps):
        # Build the inputs dict: empty for the first block, one entry for all
        # subsequent blocks in a linear pipeline (E1).  When the architecture
        # evolves to DAG (v1.0+) the executor will resolve multiple upstreams
        # here — the block.run signature does not change.
        if prev_result is None:
            inputs: dict[str, Any] = {}
        else:
            inputs = {prev_result.block_id: prev_result.data}

        try:
            result = block.run(context, inputs)
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(
                f"Block '{block.spec.block_id}' failed at step {idx + 1}/{total}: {exc}"
            ) from exc

        # Propagate metadata to context.extra for downstream blocks (ADR-014)
        context.extra["prev_metadata"] = result.metadata
        for key, value in result.metadata.items():
            context.extra[key] = value

        context.progress(block.spec.block_id, idx + 1, total)
        results.append(result)
        prev_result = result

    return results
