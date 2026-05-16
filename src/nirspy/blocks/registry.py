"""BlockRegistry — concrete implementation of RegistryProtocol.

Usage
-----
Registry instance is a module-level singleton exposed via :data:`registry`.
Blocks register themselves using the :func:`register` decorator::

    from nirspy.blocks.registry import registry

    @registry.register("load_snirf")
    class LoadSnirfBlock: ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from nirspy.domain.block import Block

_T = TypeVar("_T")


class BlockRegistry:
    """Concrete registry that maps ``block_id`` strings to :class:`~nirspy.domain.block.Block`
    instances.

    Satisfies :class:`~nirspy.domain.pipeline.RegistryProtocol` structurally.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, Block] = {}

    # ------------------------------------------------------------------
    # RegistryProtocol interface
    # ------------------------------------------------------------------

    def get(self, block_id: str) -> Block:
        """Return the block registered under *block_id*.

        Raises
        ------
        KeyError
            When *block_id* is not registered.
        """
        try:
            return self._blocks[block_id]
        except KeyError:
            registered = ", ".join(sorted(self._blocks))
            raise KeyError(
                f"Block '{block_id}' not found in registry. "
                f"Registered blocks: [{registered}]"
            ) from None

    # ------------------------------------------------------------------
    # Management helpers
    # ------------------------------------------------------------------

    def register(self, block_id: str, block: Block) -> None:
        """Register *block* under *block_id*.

        Parameters
        ----------
        block_id:
            String key used by the serialiser and pipeline loader.
        block:
            Instance of a class satisfying :class:`~nirspy.domain.block.Block`.
        """
        self._blocks[block_id] = block

    def list_blocks(self) -> list[str]:
        """Return a sorted list of all registered block IDs."""
        return sorted(self._blocks)

    def __contains__(self, block_id: object) -> bool:
        return block_id in self._blocks

    def __repr__(self) -> str:
        ids = ", ".join(self.list_blocks())
        return f"BlockRegistry([{ids}])"


# ---------------------------------------------------------------------------
# Module-level singleton + decorator helper
# ---------------------------------------------------------------------------

registry = BlockRegistry()


def register(block_id: str) -> Callable[[type[_T]], type[_T]]:
    """Class decorator that registers a block class instance in the default :data:`registry`.

    The decorated class is instantiated with no arguments and stored under *block_id*.

    Example
    -------
    ::

        @register("load_snirf")
        class LoadSnirfBlock:
            ...
    """

    def _decorator(cls: type[_T]) -> type[_T]:
        registry.register(block_id, cls())  # type: ignore[arg-type]
        return cls

    return _decorator
