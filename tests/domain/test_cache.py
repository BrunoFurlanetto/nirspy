"""Tests for CacheProtocol via InMemoryCacheAdapter (fake in-memory backend).

These tests run entirely in-process — no disk, no diskcache dependency.
The same contract is verified for DiskCacheAdapter in tests/engine/.
"""

from __future__ import annotations

import pytest

from nirspy.engine.cache_adapter import InMemoryCacheAdapter


@pytest.fixture()
def cache() -> InMemoryCacheAdapter:
    return InMemoryCacheAdapter()


class TestCacheProtocolStructural:
    """InMemoryCacheAdapter satisfies CacheProtocol structurally."""

    def test_has_required_protocol_methods(self, cache: InMemoryCacheAdapter) -> None:
        # CacheProtocol is NOT runtime_checkable; verify via hasattr.
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")
        assert hasattr(cache, "invalidate")
        assert hasattr(cache, "invalidate_from")
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

    def test_invalidate_removes_key(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("key", "value")
        cache.invalidate("key")
        assert cache.get("key") is None

    def test_invalidate_missing_key_is_noop(self, cache: InMemoryCacheAdapter) -> None:
        cache.invalidate("nonexistent")  # must not raise

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


class TestCacheInvalidateFrom:
    """invalidate_from(prefix) removes all matching keys and returns count."""

    def test_returns_zero_when_no_matches(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("alpha:1", "a")
        cache.set("alpha:2", "b")
        removed = cache.invalidate_from("beta:")
        assert removed == 0

    def test_removes_all_keys_with_prefix(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("block_a:hash1", "r1")
        cache.set("block_a:hash2", "r2")
        cache.set("block_b:hash1", "r3")
        cache.invalidate_from("block_a:")
        assert cache.get("block_a:hash1") is None
        assert cache.get("block_a:hash2") is None
        # unrelated key must be preserved
        assert cache.get("block_b:hash1") == "r3"

    def test_returns_correct_count(self, cache: InMemoryCacheAdapter) -> None:
        for i in range(5):
            cache.set(f"load_snirf:{i}", i)
        cache.set("other_block:0", "keep")
        removed = cache.invalidate_from("load_snirf:")
        assert removed == 5

    def test_empty_prefix_matches_all_keys(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        removed = cache.invalidate_from("")
        assert removed == 3

    def test_idempotent_second_call_returns_zero(self, cache: InMemoryCacheAdapter) -> None:
        cache.set("x:1", "v")
        cache.invalidate_from("x:")
        removed_again = cache.invalidate_from("x:")
        assert removed_again == 0

    def test_cascade_invalidation_pattern(self, cache: InMemoryCacheAdapter) -> None:
        """Simulates executor cascade: changing block A params invalidates A and all downstream."""
        # Populate entries for three blocks in a chain
        cache.set("block_a:abc", "out_a")
        cache.set("block_b:def", "out_b")
        cache.set("block_c:ghi", "out_c")
        # When block_a params change, invalidate block_a and everything downstream
        cache.invalidate_from("block_a:")
        cache.invalidate_from("block_b:")
        cache.invalidate_from("block_c:")
        assert cache.get("block_a:abc") is None
        assert cache.get("block_b:def") is None
        assert cache.get("block_c:ghi") is None
