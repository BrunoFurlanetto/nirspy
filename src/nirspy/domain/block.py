"""Block Protocol, BlockSpec and BlockResult — core building blocks of a pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nirspy.domain.data_types import DataType


@dataclass(frozen=True)
class BlockSpec:
    """Static descriptor for a block *type* registered in the pipeline.

    Instances are immutable and hashable — safe to use as dict keys or in sets.
    ``BlockSpec`` carries only type-level metadata; it never holds a params
    *instance*.  The concrete :class:`Block` implementation owns its own params
    instance (set at construction time), keeping ``BlockSpec`` reusable across
    multiple pipeline instances.
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
    """Optional dataclass *type* holding block parameters (ADR-007).

    Domain code stores the reference only — it never instantiates or inspects
    fields.  The GUI and registry introspect via ``dataclasses.fields(spec.params_class)``.
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

    Params strategy
    ---------------
    Each concrete block class owns its params instance as a plain attribute
    (e.g. ``self.params: LoadSnirfParams``), set at construction time by the
    registry or the caller that assembles the pipeline.  Params are **not**
    passed through :meth:`run` — the block reads them from ``self`` directly.

    This keeps :meth:`run` signature stable regardless of param shape, avoids
    threading params through :class:`~nirspy.domain.execution.ExecutionContext`,
    and lets the GUI read/write ``block.params`` via the dataclass interface
    without touching the executor.

    The domain Protocol intentionally does not declare a ``params`` attribute
    because its type varies per block type.  ``BlockSpec.params_class`` carries
    the type reference for GUI introspection.
    """

    @property
    def spec(self) -> BlockSpec:
        """Return the static type descriptor for this block."""
        ...

    def run(
        self,
        context: Any,
        inputs: dict[str, Any],
    ) -> BlockResult:
        """Execute the block logic.

        Parameters
        ----------
        context:
            :class:`~nirspy.domain.execution.ExecutionContext` injected by the
            runner.  Provides cache access and the progress callback.
        inputs:
            Mapping of ``{producer_block_id: data}`` where *data* is the
            ``BlockResult.data`` value from the upstream block.

            - For the first block in a linear pipeline the dict is empty
              (``{}``).
            - For all subsequent blocks in a linear pipeline the dict has
              exactly one key — the previous block's ``spec.block_id``.
            - When the architecture evolves to DAG (v1.0+), the dict will
              carry one entry per upstream dependency; the signature does not
              change.

        The block reads its own params from ``self`` (see *Params strategy*
        in the class docstring) — params are **not** passed here.
        """
        ...
