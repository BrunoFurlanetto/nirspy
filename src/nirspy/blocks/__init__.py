"""nirspy.blocks -- concrete pipeline blocks.

The module-level :data:`registry` is pre-populated with all built-in block
**classes** so callers only need to import this package to gain access to the
full set.  Instantiation happens at pipeline-assembly time (ADR-009).

    >>> from nirspy.blocks import registry
    >>> sorted(registry.list_blocks())
    ['bandpass_filter', 'beer_lambert', 'load_snirf', 'optical_density']
"""

from nirspy.blocks.load import LoadSnirfBlock, LoadSnirfParams
from nirspy.blocks.preprocessing import (
    BandpassFilterBlock,
    BandpassFilterParams,
    BeerLambertBlock,
    BeerLambertParams,
    OpticalDensityBlock,
    OpticalDensityParams,
)
from nirspy.blocks.registry import BlockRegistry, register, registry

# Register built-in block classes (not instances -- ADR-009)
registry.register("load_snirf", LoadSnirfBlock)
registry.register("optical_density", OpticalDensityBlock)
registry.register("beer_lambert", BeerLambertBlock)
registry.register("bandpass_filter", BandpassFilterBlock)

__all__ = [
    "BandpassFilterBlock",
    "BandpassFilterParams",
    "BeerLambertBlock",
    "BeerLambertParams",
    "BlockRegistry",
    "LoadSnirfBlock",
    "LoadSnirfParams",
    "OpticalDensityBlock",
    "OpticalDensityParams",
    "register",
    "registry",
]
