"""CacheProtocol — domain interface for result caching."""

from __future__ import annotations

from typing import Any, Protocol


class CacheProtocol(Protocol):
    """Key/value store used to cache block results across runs.

    Keys are strings; values are arbitrary objects. Concrete adapters live in
    ``nirspy.engine.cache_adapter``.

    Key convention (enforced by the executor, not this Protocol):
        ``"{block_id}:{hash_inputs}:{hash_params}"``
    """

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` if absent."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, overwriting any existing entry."""
        ...

    def invalidate(self, key: str) -> None:
        """Remove the entry for *key*. No-op if absent."""
        ...

    def invalidate_from(self, key_prefix: str) -> int:
        """Remove all entries whose key starts with *key_prefix*.

        Returns the number of entries removed.  Used by the executor to
        cascade-invalidate downstream blocks when params change.
        """
        ...

    def clear(self) -> None:
        """Remove all cached entries."""
        ...
