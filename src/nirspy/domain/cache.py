"""CacheProtocol — domain interface for result caching."""

from __future__ import annotations

from typing import Any, Protocol


class CacheProtocol(Protocol):
    """Key/value store used to cache block results across runs.

    Keys are strings; values are arbitrary objects. Concrete adapters live in
    ``nirspy.engine.cache_adapter``.
    """

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` if absent."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, overwriting any existing entry."""
        ...

    def delete(self, key: str) -> None:
        """Remove the entry for *key*. No-op if absent."""
        ...

    def clear(self) -> None:
        """Remove all cached entries."""
        ...
