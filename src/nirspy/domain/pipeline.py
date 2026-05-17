"""Pipeline dataclass and RegistryProtocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from nirspy.domain.block import Block
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.validation import validate_io_chain

_CURRENT_SCHEMA_VERSION = "0.1"


class RegistryProtocol(Protocol):
    """Lookup service that maps block IDs to :class:`~nirspy.domain.block.Block` **classes**.

    Design note (ADR-009)
    ---------------------
    The registry stores block **classes** (``type[Block]``), not instances.
    Callers are responsible for instantiating blocks with the appropriate params
    object before assembling a :class:`Pipeline`.  This design is required
    because every concrete block expects a params dataclass in its ``__init__``.

    The concrete implementation lives in ``nirspy.blocks.registry.BlockRegistry``.
    The Protocol is declared here (in ``domain/``) to avoid a circular import
    ``domain → blocks``.
    """

    def get(self, block_id: str) -> type[Block]:
        """Return the block **class** registered under *block_id*.

        Raises
        ------
        KeyError
            When *block_id* is unknown.
        """
        ...

    def register(self, block_id: str, block_cls: type[Block]) -> None:
        """Register *block_cls* under *block_id*.

        Parameters
        ----------
        block_id:
            String key used by the serialiser and pipeline loader.
        block_cls:
            A class satisfying :class:`~nirspy.domain.block.Block`.
        """
        ...

    def list_blocks(self) -> list[str]:
        """Return a sorted list of all registered block IDs."""
        ...


@dataclass
class Pipeline:
    """Ordered sequence of blocks plus their runtime parameters.

    Attributes
    ----------
    name:
        Human-readable identifier.
    steps:
        Ordered list of :class:`~nirspy.domain.block.Block` instances.
    params:
        Maps ``block_id`` → params dataclass instance (or ``None``).
    description:
        Optional free-form description.
    """

    name: str
    steps: list[Block] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the pipeline to a plain dict suitable for YAML/JSON round-trip.

        Parameters dataclasses are converted via :func:`dataclasses.asdict` when
        the IO serialiser calls this method — the serialiser is responsible for
        that conversion. Here we only store ``block_id`` references.
        """
        return {
            "schema_version": _CURRENT_SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "block_id": step.spec.block_id,
                    "enabled": step.spec.enabled,
                }
                for step in self.steps
            ],
            "params": {
                block_id: params_value
                for block_id, params_value in self.params.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], registry: RegistryProtocol) -> Pipeline:
        """Reconstruct a :class:`Pipeline` from a serialised dict.

        .. warning::

            This method performs **no-argument instantiation** of each block
            class (``block_cls()``).  It is only suitable for blocks whose
            constructor takes no required arguments (i.e. blocks without params).

            For full round-trip deserialization — including param reconstruction
            — use :func:`nirspy.io.yaml_serializer.load_pipeline`, which receives
            a :class:`~nirspy.blocks.registry.BlockRegistry`, resolves the class
            via :meth:`RegistryProtocol.get`, and instantiates each block with
            its params dataclass (ADR-009).

        Parameters
        ----------
        data:
            Plain dict produced by :meth:`to_dict` (or loaded from YAML).
        registry:
            Used to resolve ``block_id`` strings to block **classes** (not
            instances).  Each class is instantiated with no arguments.

        Raises
        ------
        ValidationError
            When ``schema_version`` is missing or incompatible, or when the
            IO type chain is broken.
        KeyError
            When a ``block_id`` is not found in *registry*.
        """
        version = data.get("schema_version")
        if version != _CURRENT_SCHEMA_VERSION:
            raise ValidationError(
                f"Unsupported pipeline schema version: {version!r}. "
                f"Expected {_CURRENT_SCHEMA_VERSION!r}."
            )

        steps_raw: list[dict[str, Any]] = data.get("steps", [])
        steps: list[Block] = []
        for entry in steps_raw:
            block_id: str = entry["block_id"]
            block_cls: type[Block] = registry.get(block_id)
            steps.append(block_cls())

        params: dict[str, Any] = data.get("params", {})

        pipeline = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            steps=steps,
            params=params,
        )

        validate_io_chain(pipeline.steps)
        return pipeline
