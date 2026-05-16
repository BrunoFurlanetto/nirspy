"""Block Protocol, BlockSpec and BlockResult — core building blocks of a pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nirspy.domain.data_types import DataType


@dataclass(frozen=True)
class BlockSpec:
    """Static descriptor for a block type registered in the pipeline.

    Instances are immutable and hashable — safe to use as dict keys or in sets.
    """

    block_id: str
    """Unique string identifier (e.g. ``"load_snirf"``)."""

    display_name: str
    """Human-readable label shown in the GUI."""

    input_type: DataType
    """Expected :class:`DataType` arriving at this block's input."""

    output_type: DataType
    """Promised :class:`DataType` produced at this block's output."""

    params_class: type[Any] | None = None
    """Optional dataclass type holding block parameters.

    Domain code stores the reference only — it never instantiates or inspects fields.
    """

    enabled: bool = True
    """Whether the block is active in the pipeline. Disabled blocks are skipped."""

    description: str = ""
    """Optional short description rendered in the GUI tooltip."""


@dataclass
class BlockResult:
    """Carries the output of a single block execution."""

    data: Any
    """The processed data object (e.g. ``mne.io.BaseRaw``)."""

    block_id: str
    """ID of the block that produced this result."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key/value pairs — blocks may store diagnostics, QC flags, etc."""


@runtime_checkable
class Block(Protocol):
    """Protocol every concrete block must satisfy.

    Implementations live in ``nirspy.blocks``. Domain code only refers to this
    Protocol — never to concrete block classes.
    """

    @property
    def spec(self) -> BlockSpec:
        """Return the static descriptor for this block."""
        ...

    def run(self, data: Any, params: Any, context: Any) -> BlockResult:
        """Execute the block logic.

        Parameters
        ----------
        data:
            Input data object (type dictated by ``spec.input_type``).
        params:
            Params dataclass instance (type dictated by ``spec.params_class``).
        context:
            :class:`~nirspy.domain.execution.ExecutionContext` — injected by the runner.
        """
        ...
