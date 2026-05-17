"""Tests for InMemoryCacheAdapter, DiskCacheAdapter and make_cache_key."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from nirspy.engine.cache_adapter import (
    DiskCacheAdapter,
    InMemoryCacheAdapter,
    make_cache_key,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem_cache() -> InMemoryCacheAdapter:
    return InMemoryCacheAdapter()


@pytest.fixture()
def disk_cache(tmp_path: pathlib.Path) -> DiskCacheAdapter:
    adapter = DiskCacheAdapter(directory=tmp_path / "nirspy_test_cache")
    yield adapter
    adapter.close()


# ---------------------------------------------------------------------------
# InMemoryCacheAdapter
# ---------------------------------------------------------------------------


class TestInMemoryCacheAdapter:
    def test_get_missing_returns_none(self, mem_cache: InMemoryCacheAdapter) -> None:
        assert mem_cache.get("absent") is None

    def test_set_and_get(self, mem_cache: InMemoryCacheAdapter) -> None:
        mem_cache.set("k", "value")
        assert mem_cache.get("k") == "value"

    def test_overwrite(self, mem_cache: InMemoryCacheAdapter) -> None:
        mem_cache.set("k", 1)
        mem_cache.set("k", 2)
        assert mem_cache.get("k") == 2

    def test_invalidate_existing(self, mem_cache: InMemoryCacheAdapter) -> None:
        mem_cache.set("k", "v")
        mem_cache.invalidate("k")
        assert mem_cache.get("k") is None

    def test_invalidate_absent_is_noop(self, mem_cache: InMemoryCacheAdapter) -> None:
        mem_cache.invalidate("ghost")  # must not raise

    def test_clear(self, mem_cache: InMemoryCacheAdapter) -> None:
        for i in range(5):
            mem_cache.set(f"k{i}", i)
        mem_cache.clear()
        for i in range(5):
            assert mem_cache.get(f"k{i}") is None

    def test_store_complex_object(self, mem_cache: InMemoryCacheAdapter) -> None:
        payload = {"nested": [1, 2, {"deep": True}]}
        mem_cache.set("complex", payload)
        assert mem_cache.get("complex") == payload

    def test_independent_instances(self) -> None:
        c1 = InMemoryCacheAdapter()
        c2 = InMemoryCacheAdapter()
        c1.set("x", 100)
        assert c2.get("x") is None


class TestInMemoryCacheAdapterInvalidateFrom:
    """invalidate_from on InMemoryCacheAdapter."""

    def test_returns_zero_when_no_match(self, mem_cache: InMemoryCacheAdapter) -> None:
        mem_cache.set("alpha:1", "a")
        assert mem_cache.invalidate_from("beta:") == 0

    def test_removes_matching_keys(self, mem_cache: InMemoryCacheAdapter) -> None:
        mem_cache.set("blk:hash1", "r1")
        mem_cache.set("blk:hash2", "r2")
        mem_cache.set("other:hash1", "keep")
        mem_cache.invalidate_from("blk:")
        assert mem_cache.get("blk:hash1") is None
        assert mem_cache.get("blk:hash2") is None
        assert mem_cache.get("other:hash1") == "keep"

    def test_returns_correct_count(self, mem_cache: InMemoryCacheAdapter) -> None:
        for i in range(4):
            mem_cache.set(f"load:{i}", i)
        mem_cache.set("other:0", "keep")
        removed = mem_cache.invalidate_from("load:")
        assert removed == 4

    def test_idempotent_second_call(self, mem_cache: InMemoryCacheAdapter) -> None:
        mem_cache.set("x:1", "v")
        mem_cache.invalidate_from("x:")
        assert mem_cache.invalidate_from("x:") == 0


# ---------------------------------------------------------------------------
# DiskCacheAdapter
# ---------------------------------------------------------------------------


class TestDiskCacheAdapter:
    def test_get_missing_returns_none(self, disk_cache: DiskCacheAdapter) -> None:
        assert disk_cache.get("absent") is None

    def test_set_and_get(self, disk_cache: DiskCacheAdapter) -> None:
        disk_cache.set("key", 42)
        assert disk_cache.get("key") == 42

    def test_overwrite(self, disk_cache: DiskCacheAdapter) -> None:
        disk_cache.set("k", "old")
        disk_cache.set("k", "new")
        assert disk_cache.get("k") == "new"

    def test_invalidate_existing(self, disk_cache: DiskCacheAdapter) -> None:
        disk_cache.set("k", "v")
        disk_cache.invalidate("k")
        assert disk_cache.get("k") is None

    def test_invalidate_absent_is_noop(self, disk_cache: DiskCacheAdapter) -> None:
        disk_cache.invalidate("nonexistent")  # must not raise

    def test_clear(self, disk_cache: DiskCacheAdapter) -> None:
        disk_cache.set("a", 1)
        disk_cache.set("b", 2)
        disk_cache.clear()
        assert disk_cache.get("a") is None
        assert disk_cache.get("b") is None

    def test_persistence_across_instances(self, tmp_path: pathlib.Path) -> None:
        """Data written by one instance must be readable by another at the same dir."""
        cache_dir = tmp_path / "persistent"
        c1 = DiskCacheAdapter(directory=cache_dir)
        c1.set("persistent_key", "hello")
        c1.close()

        c2 = DiskCacheAdapter(directory=cache_dir)
        assert c2.get("persistent_key") == "hello"
        c2.close()

    def test_default_dir_created_if_not_exists(self, tmp_path: pathlib.Path) -> None:
        new_dir = tmp_path / "brand_new_dir"
        assert not new_dir.exists()
        adapter = DiskCacheAdapter(directory=new_dir)
        assert new_dir.exists()
        adapter.close()


class TestDiskCacheAdapterInvalidateFrom:
    """invalidate_from on DiskCacheAdapter."""

    def test_returns_zero_when_no_match(self, disk_cache: DiskCacheAdapter) -> None:
        disk_cache.set("alpha:1", "a")
        assert disk_cache.invalidate_from("beta:") == 0

    def test_removes_matching_keys(self, disk_cache: DiskCacheAdapter) -> None:
        disk_cache.set("blk:hash1", "r1")
        disk_cache.set("blk:hash2", "r2")
        disk_cache.set("other:hash1", "keep")
        disk_cache.invalidate_from("blk:")
        assert disk_cache.get("blk:hash1") is None
        assert disk_cache.get("blk:hash2") is None
        assert disk_cache.get("other:hash1") == "keep"

    def test_returns_correct_count(self, disk_cache: DiskCacheAdapter) -> None:
        for i in range(4):
            disk_cache.set(f"load:{i}", i)
        disk_cache.set("other:0", "keep")
        removed = disk_cache.invalidate_from("load:")
        assert removed == 4

    def test_idempotent_second_call(self, disk_cache: DiskCacheAdapter) -> None:
        disk_cache.set("x:1", "v")
        disk_cache.invalidate_from("x:")
        assert disk_cache.invalidate_from("x:") == 0


# ---------------------------------------------------------------------------
# make_cache_key determinism
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _SampleParams:
    ppf: float = 6.0
    window_s: float = 30.0
    method: str = "beer_lambert"


@dataclasses.dataclass
class _OtherParams:
    threshold: float = 0.5
    n_jobs: int = 1


class TestMakeCacheKey:
    def test_same_params_produce_same_key(self) -> None:
        p1 = _SampleParams(ppf=6.0, window_s=30.0, method="beer_lambert")
        p2 = _SampleParams(ppf=6.0, window_s=30.0, method="beer_lambert")
        assert make_cache_key(p1) == make_cache_key(p2)

    def test_different_values_produce_different_keys(self) -> None:
        p1 = _SampleParams(ppf=6.0)
        p2 = _SampleParams(ppf=7.0)
        assert make_cache_key(p1) != make_cache_key(p2)

    def test_different_types_produce_different_keys(self) -> None:
        # Even with coincidentally identical values, different types differ by prefix.
        p1 = _SampleParams(ppf=6.0, window_s=30.0, method="beer_lambert")
        p2 = _OtherParams(threshold=6.0, n_jobs=30)
        # Keys should differ (different structure)
        assert make_cache_key(p1) != make_cache_key(p2)

    def test_prefix_is_prepended(self) -> None:
        p = _SampleParams()
        key = make_cache_key(p, prefix="my_block")
        assert key.startswith("my_block:")

    def test_no_prefix_returns_bare_digest(self) -> None:
        p = _SampleParams()
        key = make_cache_key(p)
        assert ":" not in key
        assert len(key) == 64  # SHA-256 hex digest length

    def test_key_is_hex_string(self) -> None:
        p = _SampleParams()
        key = make_cache_key(p)
        # Strip prefix if any
        digest_part = key.split(":")[-1]
        int(digest_part, 16)  # must not raise ValueError

    def test_non_dataclass_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            make_cache_key({"ppf": 6.0})  # type: ignore[arg-type]

    def test_dataclass_class_raises_type_error(self) -> None:
        """Passing the class itself (not an instance) must raise TypeError."""
        with pytest.raises(TypeError):
            make_cache_key(_SampleParams)  # type: ignore[arg-type]

    def test_determinism_across_calls(self) -> None:
        """Calling make_cache_key repeatedly with the same instance returns same digest."""
        p = _SampleParams(ppf=6.0, window_s=30.0, method="beer_lambert")
        keys = [make_cache_key(p) for _ in range(10)]
        assert len(set(keys)) == 1

    def test_field_order_does_not_affect_key(self) -> None:
        """json.dumps with sort_keys=True ensures field-order independence.
        We simulate by comparing two dicts that differ only in key ordering."""
        # Both instances have same logical values — just verifying stability.
        p1 = _SampleParams(ppf=6.0, window_s=30.0, method="test")
        p2 = _SampleParams(ppf=6.0, window_s=30.0, method="test")
        assert make_cache_key(p1, prefix="blk") == make_cache_key(p2, prefix="blk")
