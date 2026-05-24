"""nirspy.blocks -- concrete pipeline blocks.

The module-level :data:`registry` is pre-populated with all built-in block
**classes** so callers only need to import this package to gain access to the
full set.  Instantiation happens at pipeline-assembly time (ADR-009).

    >>> from nirspy.blocks import registry
    >>> sorted(registry.list_blocks())
    ['bandpass_filter', 'beer_lambert', 'block_average', 'glm', 'load_snirf',
     'manual_channel_exclude', 'optical_density', 'prune_channels',
     'scalp_coupling_index', 'spline_motion_correction', 'tddr',
     'wavelet_motion_correction']
"""

from nirspy.blocks.analysis import (
    BlockAverageBlock,
    BlockAverageParams,
    ConditionWindow,
)
from nirspy.blocks.epochs import (
    EpochsExtractionBlock,
    EpochsExtractionParams,
)
from nirspy.blocks.export import (
    ExportTableBlock,
    ExportTableParams,
)
from nirspy.blocks.glm import (
    GLMBlock,
    GLMParams,
)
from nirspy.blocks.load import LoadSnirfBlock, LoadSnirfParams
from nirspy.blocks.manual_exclude import (
    ManualChannelExcludeBlock,
    ManualChannelExcludeParams,
)
from nirspy.blocks.motion import (
    SplineBlock,
    SplineParams,
    TDDRBlock,
    TDDRParams,
    WaveletBlock,
    WaveletParams,
)
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
from nirspy.blocks.signal_enhancement import (
    ShortChannelRegressionBlock,
    ShortChannelRegressionParams,
)

# Register built-in block classes (not instances -- ADR-009)
registry.register("load_snirf", LoadSnirfBlock)
registry.register("optical_density", OpticalDensityBlock)
registry.register("beer_lambert", BeerLambertBlock)
registry.register("bandpass_filter", BandpassFilterBlock)
registry.register("tddr", TDDRBlock)
registry.register("spline_motion_correction", SplineBlock)
registry.register("wavelet_motion_correction", WaveletBlock)
registry.register("scalp_coupling_index", ScalpCouplingIndexBlock)
registry.register("prune_channels", PruneChannelsBlock)
registry.register("block_average", BlockAverageBlock)
registry.register("manual_channel_exclude", ManualChannelExcludeBlock)
registry.register("short_channel_regression", ShortChannelRegressionBlock)
registry.register("epochs_extraction", EpochsExtractionBlock)
registry.register("export_table", ExportTableBlock)
registry.register("glm", GLMBlock)

__all__ = [
    "BandpassFilterBlock",
    "BandpassFilterParams",
    "BeerLambertBlock",
    "BeerLambertParams",
    "BlockAverageBlock",
    "BlockAverageParams",
    "ConditionWindow",
    "BlockRegistry",
    "EpochsExtractionBlock",
    "EpochsExtractionParams",
    "ExportTableBlock",
    "ExportTableParams",
    "GLMBlock",
    "GLMParams",
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
    "ShortChannelRegressionBlock",
    "ShortChannelRegressionParams",
    "SplineBlock",
    "SplineParams",
    "TDDRBlock",
    "TDDRParams",
    "WaveletBlock",
    "WaveletParams",
    "register",
    "registry",
]
