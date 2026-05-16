"""Pipeline dataclass and RegistryProtocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from nirspy.domain.block import Block
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.validation import validate_io_chain

_CURRENT_SCHEMA_VERSION = "0.1"


class RegistryProtocol(Protocol):
    """Lookup service that maps block IDs to :class:`~nirspy.domain.block.Block` instances."""

    def get(self, block_id: str) -> Block:
        """Return the block registered under *block_id*.

        Raises
        ------
        KeyError
            When *block_id* is unknown.
        """
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

        Parameters
        ----------
        data:
            Plain dict produced by :meth:`to_dict` (or loaded from YAML).
        registry:
            Used to resolve ``block_id`` strings back to :class:`Block` objects.

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
            steps.append(registry.get(block_id))

        params: dict[str, Any] = data.get("params", {})

        pipeline = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            steps=steps,
            params=params,
        )

        validate_io_chain(pipeline.steps)
        return pipeline
