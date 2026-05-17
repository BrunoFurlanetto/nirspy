"""Concrete cache adapters implementing :class:`~nirspy.domain.cache.CacheProtocol`.

Provides:
- ``InMemoryCacheAdapter`` -- plain dict, no persistence (default for tests/dev).
- ``DiskCacheAdapter`` -- persistent cache backed by ``diskcache.Cache``.

Security (S-01):
    DiskCacheAdapter uses a custom JSONDisk that serializes with JSON instead of
    Python pickle, eliminating the RCE vector from arbitrary deserialization.
    Only JSON-safe types are cached; numpy arrays are stored as nested lists.

Hash utility
------------
Both adapters expose the module-level :func:`make_cache_key` helper that converts
a params dataclass into a deterministic SHA-256 hex digest.
The hash uses ``json.dumps(asdict(params), sort_keys=True)`` so the result is
identical across machines and Python versions (D5 decision).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache
import numpy as np

_DEFAULT_CACHE_DIR = Path.home() / ".nirspy" / "cache"

# ---------------------------------------------------------------------------
# JSON serialization helpers (S-01: replaces pickle)
# ---------------------------------------------------------------------------

_NUMPY_MARKER = "__nirspy_ndarray__"


class _NirspyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and basic numpy scalars."""

    def default(self, o: Any) -> Any:  # noqa: ANN401
        if isinstance(o, np.ndarray):
            return {
                _NUMPY_MARKER: True,
                "dtype": str(o.dtype),
                "shape": list(o.shape),
                "data": o.tolist(),
            }
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def _nirspy_decoder(obj: dict[str, Any]) -> Any:
    """JSON object_hook that reconstructs numpy arrays."""
    if _NUMPY_MARKER in obj:
        return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


def _serialize_value(value: Any) -> bytes:
    """Serialize a cache value to JSON bytes.

    Raises TypeError if the value contains non-serializable objects (e.g. MNE Raw).
    This is intentional -- only safe, reproducible data should be cached.
    """
    return json.dumps(value, cls=_NirspyEncoder, sort_keys=True).encode("utf-8")


def _deserialize_value(data: bytes) -> Any:
    """Deserialize JSON bytes back to a Python object."""
    return json.loads(data.decode("utf-8"), object_hook=_nirspy_decoder)

# ---------------------------------------------------------------------------
# Custom diskcache Disk class (S-01: no pickle)
# ---------------------------------------------------------------------------


class JSONDisk(diskcache.Disk):  # type: ignore[misc]
    """Disk backend that uses JSON instead of pickle for value serialization.

    Keys remain as default (strings). Values are serialized via JSON with
    numpy array support. Non-JSON-serializable values raise TypeError at
    write time rather than silently pickling.
    """

    def store(  # type: ignore[override]
        self,
        value: Any,
        read: bool,
        key: Any = diskcache.UNKNOWN,
    ) -> tuple[int, int, str | None, bytes]:
        """Serialize value to JSON bytes then delegate storage to base Disk."""
        if not read:
            value = _serialize_value(value)
        return super().store(value, read, key=key)  # type: ignore[no-any-return]

    def fetch(  # type: ignore[override]
        self,
        mode: int,
        filename: str | None,
        value: Any,
        read: bool,
    ) -> Any:
        """Delegate fetch to base Disk then deserialize JSON bytes."""
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = _deserialize_value(data)
        return data

# ---------------------------------------------------------------------------
# Hash utility
# ---------------------------------------------------------------------------


def make_cache_key(params: Any, prefix: str = "") -> str:
    """Return a deterministic SHA-256 hex digest for *params*.

    Parameters
    ----------
    params:
        A dataclass instance. Must be convertible via :func:`dataclasses.asdict`.
    prefix:
        Optional string prepended to the digest (e.g. the block ID) to avoid
        cross-block collisions when params happen to be identical.

    Raises
    ------
    TypeError
        When *params* is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(params) or isinstance(params, type):
        raise TypeError(f"params must be a dataclass instance, got {type(params)!r}")

    canonical = json.dumps(dataclasses.asdict(params), sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"{prefix}:{digest}" if prefix else digest

# ---------------------------------------------------------------------------
# InMemoryCacheAdapter
# ---------------------------------------------------------------------------


class InMemoryCacheAdapter:
    """Volatile in-process cache backed by a plain :class:`dict`.

    Not thread-safe -- sufficient for Etapa 1 single-threaded runner.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def invalidate(self, key: str) -> None:
        """Remove the entry for *key*. No-op if absent."""
        self._store.pop(key, None)

    def invalidate_from(self, key_prefix: str) -> int:
        """Remove all entries whose key starts with *key_prefix*.

        Returns the number of entries removed.
        """
        matching = [k for k in self._store if k.startswith(key_prefix)]
        for k in matching:
            del self._store[k]
        return len(matching)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# DiskCacheAdapter (S-01: uses JSONDisk, no pickle)
# ---------------------------------------------------------------------------


class DiskCacheAdapter:
    """Persistent cache backed by :class:`diskcache.Cache` with JSON serialization.

    Security (S-01): Uses :class:`JSONDisk` instead of the default pickle-based
    Disk, eliminating the RCE vector from arbitrary deserialization of cached
    data. Only JSON-safe types (including numpy arrays) can be cached.

    Parameters
    ----------
    directory:
        Path to the cache directory. Defaults to ``~/.nirspy/cache``.
    """

    def __init__(self, directory: Path | None = None) -> None:
        cache_dir = directory if directory is not None else _DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: diskcache.Cache = diskcache.Cache(
            str(cache_dir),
            disk=JSONDisk,
        )

    def get(self, key: str) -> Any | None:
        return self._cache.get(key, default=None)

    def set(self, key: str, value: Any) -> None:
        self._cache.set(key, value)

    def invalidate(self, key: str) -> None:
        """Remove the entry for *key*. No-op if absent."""
        self._cache.delete(key)

    def invalidate_from(self, key_prefix: str) -> int:
        """Remove all entries whose key starts with *key_prefix*.

        Returns the number of entries removed.

        Complexity: O(n_cached_keys) -- acceptable for E1 pipeline sizes.
        """
        matching = [k for k in self._cache if isinstance(k, str) and k.startswith(key_prefix)]
        for k in matching:
            self._cache.delete(k)
        return len(matching)

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        """Flush and close the underlying diskcache handle."""
        self._cache.close()
