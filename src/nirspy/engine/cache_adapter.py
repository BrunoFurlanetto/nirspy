"""Concrete cache adapters implementing :class:`~nirspy.domain.cache.CacheProtocol`.

Provides:
- ``InMemoryCacheAdapter`` — plain dict, no persistence (default for tests/dev).
- ``DiskCacheAdapter`` — persistent cache backed by ``diskcache.Cache``.

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

_DEFAULT_CACHE_DIR = Path.home() / ".nirspy" / "cache"


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


class InMemoryCacheAdapter:
    """Volatile in-process cache backed by a plain :class:`dict`.

    Not thread-safe — sufficient for Etapa 1 single-threaded runner.
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


class DiskCacheAdapter:
    """Persistent cache backed by :class:`diskcache.Cache`.

    Parameters
    ----------
    directory:
        Path to the cache directory. Defaults to ``~/.nirspy/cache``.
    """

    def __init__(self, directory: Path | None = None) -> None:
        cache_dir = directory if directory is not None else _DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: diskcache.Cache = diskcache.Cache(str(cache_dir))

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

        Complexity: O(n_cached_keys) — acceptable for E1 pipeline sizes.
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
