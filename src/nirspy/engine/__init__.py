"""nirspy.engine — MNE-NIRS adapter layer."""

from nirspy.engine.cache_adapter import DiskCacheAdapter, InMemoryCacheAdapter, make_cache_key
from nirspy.engine.exceptions import (
    UI_ERROR_MESSAGES,
    AdapterError,
    EngineError,
    MNEOperationError,
    SnirfLoadError,
)
from nirspy.engine.mne_adapter import MNEAdapter, RawWrapper

__all__ = [
    "AdapterError",
    "DiskCacheAdapter",
    "EngineError",
    "InMemoryCacheAdapter",
    "MNEAdapter",
    "MNEOperationError",
    "RawWrapper",
    "SnirfLoadError",
    "UI_ERROR_MESSAGES",
    "make_cache_key",
]
