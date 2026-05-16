"""nirspy.engine — MNE-NIRS adapter layer."""

from nirspy.engine.cache_adapter import DiskCacheAdapter, InMemoryCacheAdapter, make_cache_key
from nirspy.engine.exceptions import AdapterError, EngineError, SnirfLoadError
from nirspy.engine.mne_adapter import MNEAdapter, RawWrapper

__all__ = [
    "AdapterError",
    "DiskCacheAdapter",
    "EngineError",
    "InMemoryCacheAdapter",
    "MNEAdapter",
    "RawWrapper",
    "SnirfLoadError",
    "make_cache_key",
]
