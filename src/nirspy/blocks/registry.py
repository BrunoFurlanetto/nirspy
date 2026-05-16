"""BlockRegistry — concrete implementation of RegistryProtocol.

Usage
-----
Registry instance is a module-level singleton exposed via :data:`registry`.
Blocks register themselves using the :func:`register` decorator::

    from nirspy.blocks.registry import registry

    @registry.register("load_snirf")
    class LoadSnirfBlock: ...

Design note (ADR-009)
---------------------
The registry stores **classes** (``type[Block]``), not instances.  Callers are
responsible for instantiating blocks with the appropriate params object before
assembling a :class:`~nirspy.domain.pipeline.Pipeline`.  This model is
required because every concrete block expects a params dataclass in its
``__init__`` — a no-argument instantiation is no longer possible.

The :class:`~nirspy.domain.pipeline.RegistryProtocol` declared in the domain
layer expects ``get() -> Block`` (an instance), which is incompatible with the
new class-based storage.  For Etapa 1, :func:`~nirspy.io.yaml_serializer`
accepts a ``BlockRegistry`` directly (instead of the Protocol) and calls
:meth:`get` to obtain the class, then instantiates it with the deserialized
params.  The Protocol will be revisited when the domain executor needs to
perform its own class resolution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from nirspy.domain.block import Block

_T = TypeVar("_T")


class BlockRegistry:
    """Concrete registry that maps ``block_id`` strings to :class:`~nirspy.domain.block.Block`
    **classes** (not instances).

    Each call to :meth:`get` returns the class so callers can instantiate it
    with the appropriate ``params`` object.
    """

    def __init__(self) -> None:
        self._classes: dict[str, type[Block]] = {}

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def get(self, block_id: str) -> type[Block]:
        """Return the block **class** registered under *block_id*.

        Raises
        ------
        KeyError
            When *block_id* is not registered.
        """
        try:
            return self._classes[block_id]
        except KeyError:
            registered = ", ".join(sorted(self._classes))
            raise KeyError(
                f"Block '{block_id}' not found in registry. "
                f"Registered blocks: [{registered}]"
            ) from None

    # ------------------------------------------------------------------
    # Management helpers
    # ------------------------------------------------------------------

    def register(self, block_id: str, block_cls: type[Block]) -> None:
        """Register *block_cls* under *block_id*.

        Parameters
        ----------
        block_id:
            String key used by the serialiser and pipeline loader.
        block_cls:
            A class satisfying :class:`~nirspy.domain.block.Block`.
        """
        self._classes[block_id] = block_cls

    def list_blocks(self) -> list[str]:
        """Return a sorted list of all registered block IDs."""
        return sorted(self._classes)

    def __contains__(self, block_id: object) -> bool:
        return block_id in self._classes

    def __repr__(self) -> str:
        ids = ", ".join(self.list_blocks())
        return f"BlockRegistry([{ids}])"


# ---------------------------------------------------------------------------
# Module-level singleton + decorator helper
# ---------------------------------------------------------------------------

registry = BlockRegistry()


def register(block_id: str) -> Callable[[type[_T]], type[_T]]:
    """Class decorator that registers a block **class** in the default :data:`registry`.

    Unlike the previous implementation, the class is stored directly — no
    instantiation happens at decoration time, consistent with ADR-009.

    Example
    -------
    ::

        @register("load_snirf")
        class LoadSnirfBlock:
            ...
    """

    def _decorator(cls: type[_T]) -> type[_T]:
        registry.register(block_id, cls)  # type: ignore[arg-type]
        return cls

    return _decorator
