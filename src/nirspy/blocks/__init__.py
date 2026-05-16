"""nirspy.blocks — concrete pipeline blocks.

The module-level :data:`registry` is pre-populated with all built-in block
**classes** so callers only need to import this package to gain access to the
full set.  Instantiation happens at pipeline-assembly time (ADR-009).

    >>> from nirspy.blocks import registry
    >>> registry.list_blocks()
    ['load_snirf']
"""

from nirspy.blocks.load import LoadSnirfBlock, LoadSnirfParams
from nirspy.blocks.registry import BlockRegistry, register, registry

# Register built-in block classes (not instances — ADR-009)
registry.register("load_snirf", LoadSnirfBlock)

__all__ = [
    "BlockRegistry",
    "LoadSnirfBlock",
    "LoadSnirfParams",
    "register",
    "registry",
]
