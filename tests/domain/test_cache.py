"""Tests for CacheProtocol via InMemoryCacheAdapter (fake in-memory backend).

These tests run entirely in-process — no disk, no diskcache dependency.
The same contract is verified for DiskCacheAdapter in tests/engine/.
"""

from __future__ import annotations

import pytest

from nirspy.domain.cache import CacheProtocol
from nirspy.engine.cache_adapter import InMemoryCacheAdapter


@pytest.fixture()
def cache() -> InMemoryCacheAdapter:
    return InMemoryCacheAdapter()


class TestCacheProtocolStructural:
    """InMemoryCacheAdapter satisfies CacheProtocol structurally."""

    def test_is_instance_of_cache_protocol(self, cache: InMemoryCacheAdapter) -> None:
        # CacheProtocol is NOT runtime_checkable in the current implementation,
        # so we verify structural compliance by checking method existence.
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")
        assert hasattr(cache, "delete")
        assert hasattr(cache, "clear")

    def test_get_returns_none_for_missing_key(self, cache: InMemoryCacheAdapter) -> None:
        assert cache.get("missing") is None

    def test_set_and_get_roundtrip(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("key1", 42)
        assert cache.get("key1") == 42

    def test_set_overwrites_existing(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("k", "first")
        cache.set("k", "second")
        assert cache.get("k") == "second"

    def test_delete_removes_key(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("key", "value")
        cache.delete("key")
        assert cache.get("key") is None

    def test_delete_missing_key_is_noop(self, cache: InMemoryCacheAdapter) -> None:
        cache.delete("nonexistent")  # must not raise

    def test_clear_removes_all_entries(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") is None

    def test_clear_on_empty_is_noop(self, cache: InMemoryCacheAdapter) -> None:
        cache.clear()  # must not raise

    def test_can_store_any_value_type(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"x": 1})
        cache.set("none", None)
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"x": 1}
        assert cache.get("none") is None

    def test_independent_instances_do_not_share_state(self) -> None:
        c1 = InMemoryCacheAdapter()
        c2 = InMemoryCacheAdapter()
        c1.set("shared_key", "from_c1")
        assert c2.get("shared_key") is None

    def test_many_keys(self, cache: InMemoryCacheAdapter) -> None:
        for i in range(100):
            cache.set(f"key_{i}", i)
        for i in range(100):
            assert cache.get(f"key_{i}") == i
