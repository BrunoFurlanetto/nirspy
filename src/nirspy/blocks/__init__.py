"""nirspy.blocks -- concrete pipeline blocks.

The module-level :data:`registry` is pre-populated with all built-in block
**classes** so callers only need to import this package to gain access to the
full set.  Instantiation happens at pipeline-assembly time (ADR-009).

    >>> from nirspy.blocks import registry
    >>> sorted(registry.list_blocks())
    ['bandpass_filter', 'beer_lambert', 'block_average', 'load_snirf',
     'manual_channel_exclude', 'optical_density', 'prune_channels',
     'scalp_coupling_index', 'tddr']
"""

from nirspy.blocks.analysis import (
    BlockAverageBlock,
    BlockAverageParams,
    ConditionWindow,
)
from nirspy.blocks.load import LoadSnirfBlock, LoadSnirfParams
from nirspy.blocks.manual_exclude import (
    ManualChannelExcludeBlock,
    ManualChannelExcludeParams,
)
from nirspy.blocks.motion import TDDRBlock, TDDRParams
from nirspy.blocks.preprocessing import (
    BandpassFilterBlock,
    BandpassFilterParams,
    BeerLambertBlock,
    BeerLambertParams,
    OpticalDensityBlock,
    OpticalDensityParams,
)
from nirspy.blocks.quality import (
    PruneChannelsBlock,
    PruneChannelsParams,
    ScalpCouplingIndexBlock,
    ScalpCouplingIndexParams,
)
from nirspy.blocks.registry import BlockRegistry, register, registry

# Register built-in block classes (not instances -- ADR-009)
registry.register("load_snirf", LoadSnirfBlock)
registry.register("optical_density", OpticalDensityBlock)
registry.register("beer_lambert", BeerLambertBlock)
registry.register("bandpass_filter", BandpassFilterBlock)
registry.register("tddr", TDDRBlock)
registry.register("scalp_coupling_index", ScalpCouplingIndexBlock)
registry.register("prune_channels", PruneChannelsBlock)
registry.register("block_average", BlockAverageBlock)
registry.register("manual_channel_exclude", ManualChannelExcludeBlock)

__all__ = [
    "BandpassFilterBlock",
    "BandpassFilterParams",
    "BeerLambertBlock",
    "BeerLambertParams",
    "BlockAverageBlock",
    "BlockAverageParams",
    "ConditionWindow",
    "BlockRegistry",
    "ManualChannelExcludeBlock",
    "ManualChannelExcludeParams",
    "LoadSnirfBlock",
    "LoadSnirfParams",
    "OpticalDensityBlock",
    "OpticalDensityParams",
    "PruneChannelsBlock",
    "PruneChannelsParams",
    "ScalpCouplingIndexBlock",
    "ScalpCouplingIndexParams",
    "TDDRBlock",
    "TDDRParams",
    "register",
    "registry",
]
