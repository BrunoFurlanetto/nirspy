"""nirspy.blocks — concrete pipeline blocks.

The module-level :data:`registry` is pre-populated with all built-in blocks
so callers only need to import this package to gain access to the full set.

    >>> from nirspy.blocks import registry
    >>> registry.list_blocks()
    ['load_snirf']
"""

from nirspy.blocks.load import LoadSnirfBlock, LoadSnirfParams
from nirspy.blocks.registry import BlockRegistry, register, registry

# Register built-in blocks
registry.register("load_snirf", LoadSnirfBlock())

__all__ = [
    "BlockRegistry",
    "LoadSnirfBlock",
    "LoadSnirfParams",
    "register",
    "registry",
]
