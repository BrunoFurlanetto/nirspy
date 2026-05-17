"""Tests for Pipeline dataclass — construction, to_dict/from_dict round-trip."""

from __future__ import annotations

from typing import Any

import pytest

from nirspy.domain.block import Block
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.pipeline import _CURRENT_SCHEMA_VERSION, Pipeline
from tests.conftest import make_block

# ---------------------------------------------------------------------------
# Fake RegistryProtocol implementation for tests
# ---------------------------------------------------------------------------


def _block_class_from_instance(block: Block) -> type[Block]:
    """Return a no-arg-constructible class whose instances carry *block*'s spec.

    ``Pipeline.from_dict`` calls ``block_cls()`` (no arguments) on whatever the
    registry returns.  ``_FakeBlockImpl`` requires ``_spec`` at construction time,
    so we synthesise a thin subclass that bakes the spec in via a default argument.
    """
    from tests.conftest import _FakeBlockImpl  # local import to avoid circular ref

    spec = block.spec

    class _BoundFakeBlock(_FakeBlockImpl):
        def __init__(self) -> None:  # type: ignore[override]
            super().__init__(_spec=spec)

    _BoundFakeBlock.__name__ = f"FakeBlock_{spec.block_id}"
    _BoundFakeBlock.__qualname__ = f"FakeBlock_{spec.block_id}"
    return _BoundFakeBlock  # type: ignore[return-value]


class FakeRegistry:
    """In-memory registry satisfying RegistryProtocol.

    Stores block **classes** (``type[Block]``), not instances, mirroring the
    post-ADR-009 ``RegistryProtocol`` contract.
    """

    def __init__(self, blocks: dict[str, type[Block]]) -> None:
        self._blocks = blocks

    def get(self, block_id: str) -> type[Block]:
        if block_id not in self._blocks:
            raise KeyError(f"Unknown block_id: {block_id!r}")
        return self._blocks[block_id]

    def register(self, block_id: str, block_cls: type[Block]) -> None:
        self._blocks[block_id] = block_cls

    def list_blocks(self) -> list[str]:
        return sorted(self._blocks)


def _make_registry(*blocks: Block) -> FakeRegistry:
    """Build a FakeRegistry from existing Block *instances*.

    Each instance is converted to a no-arg-constructible class so that
    ``Pipeline.from_dict`` can call ``block_cls()`` successfully.
    """
    return FakeRegistry({b.spec.block_id: _block_class_from_instance(b) for b in blocks})


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    def test_minimal_pipeline(self) -> None:
        p = Pipeline(name="empty")
        assert p.name == "empty"
        assert p.steps == []
        assert p.params == {}
        assert p.description == ""

    def test_pipeline_with_steps(self) -> None:
        b1 = make_block("raw_loader", DataType.ANY, DataType.RAW)
        b2 = make_block("raw_to_od", DataType.RAW, DataType.RAW_OD)
        p = Pipeline(name="test", steps=[b1, b2])
        assert len(p.steps) == 2

    def test_pipeline_with_description(self) -> None:
        p = Pipeline(name="my_pipeline", description="for testing")
        assert "testing" in p.description


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_current_schema_version_is_string(self) -> None:
        assert isinstance(_CURRENT_SCHEMA_VERSION, str)
        assert _CURRENT_SCHEMA_VERSION  # not empty

    def test_to_dict_includes_schema_version(self) -> None:
        p = Pipeline(name="v_test")
        d = p.to_dict()
        assert d["schema_version"] == _CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestPipelineToDict:
    def test_empty_pipeline_to_dict(self) -> None:
        p = Pipeline(name="empty", description="nothing")
        d = p.to_dict()
        assert d["name"] == "empty"
        assert d["description"] == "nothing"
        assert d["steps"] == []
        assert "schema_version" in d

    def test_steps_serialised_as_list_of_block_ids(self) -> None:
        b1 = make_block("load", DataType.ANY, DataType.RAW)
        b2 = make_block("od", DataType.RAW, DataType.RAW_OD)
        p = Pipeline(name="two_step", steps=[b1, b2])
        d = p.to_dict()
        ids = [s["block_id"] for s in d["steps"]]
        assert ids == ["load", "od"]

    def test_enabled_flag_preserved(self) -> None:
        b_enabled = make_block("b1", DataType.RAW, DataType.RAW_OD, enabled=True)
        b_disabled = make_block("b2", DataType.RAW_OD, DataType.RAW_HAEMO, enabled=False)
        p = Pipeline(name="mixed", steps=[b_enabled, b_disabled])
        d = p.to_dict()
        assert d["steps"][0]["enabled"] is True
        assert d["steps"][1]["enabled"] is False

    def test_params_included_in_dict(self) -> None:
        p = Pipeline(name="p", params={"load": {"path": "/data/file.snirf"}})
        d = p.to_dict()
        assert d["params"]["load"]["path"] == "/data/file.snirf"


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


class TestPipelineFromDict:
    def _make_valid_dict(self, *block_ids: str) -> dict[str, Any]:
        return {
            "schema_version": _CURRENT_SCHEMA_VERSION,
            "name": "test",
            "description": "",
            "steps": [{"block_id": bid, "enabled": True} for bid in block_ids],
            "params": {},
        }

    def test_empty_pipeline_roundtrip(self) -> None:
        p_orig = Pipeline(name="empty")
        d = p_orig.to_dict()
        p_loaded = Pipeline.from_dict(d, FakeRegistry({}))
        assert p_loaded.name == "empty"
        assert p_loaded.steps == []

    def test_from_dict_reconstructs_blocks(self) -> None:
        b1 = make_block("load", DataType.ANY, DataType.RAW)
        b2 = make_block("od", DataType.RAW, DataType.RAW_OD)
        registry = _make_registry(b1, b2)
        data = self._make_valid_dict("load", "od")
        p = Pipeline.from_dict(data, registry)
        assert len(p.steps) == 2
        assert p.steps[0].spec.block_id == "load"
        assert p.steps[1].spec.block_id == "od"

    def test_wrong_schema_version_raises(self) -> None:
        data = self._make_valid_dict()
        data["schema_version"] = "99.99"
        with pytest.raises(ValidationError, match="schema"):
            Pipeline.from_dict(data, FakeRegistry({}))

    def test_missing_schema_version_raises(self) -> None:
        data = self._make_valid_dict()
        del data["schema_version"]
        with pytest.raises(ValidationError):
            Pipeline.from_dict(data, FakeRegistry({}))

    def test_unknown_block_id_raises_key_error(self) -> None:
        data = self._make_valid_dict("nonexistent_block")
        with pytest.raises(KeyError):
            Pipeline.from_dict(data, FakeRegistry({}))

    def test_incompatible_io_chain_raises_validation_error(self) -> None:
        b1 = make_block("load", DataType.ANY, DataType.RAW_OD)
        b2 = make_block("haemo", DataType.EPOCHS, DataType.EVOKED)
        registry = _make_registry(b1, b2)
        data = self._make_valid_dict("load", "haemo")
        with pytest.raises(ValidationError):
            Pipeline.from_dict(data, registry)

    def test_params_restored(self) -> None:
        registry = FakeRegistry({})
        data = {
            "schema_version": _CURRENT_SCHEMA_VERSION,
            "name": "with_params",
            "description": "",
            "steps": [],
            "params": {"load": {"path": "/foo.snirf"}},
        }
        p = Pipeline.from_dict(data, registry)
        assert p.params["load"]["path"] == "/foo.snirf"


# ---------------------------------------------------------------------------
# Round-trip golden test
# ---------------------------------------------------------------------------


class TestPipelineRoundTrip:
    """to_dict → from_dict → to_dict must produce identical dicts."""

    def test_three_block_roundtrip(self) -> None:
        b1 = make_block("load", DataType.ANY, DataType.RAW)
        b2 = make_block("od", DataType.RAW, DataType.RAW_OD)
        b3 = make_block("haemo", DataType.RAW_OD, DataType.RAW_HAEMO)
        registry = _make_registry(b1, b2, b3)

        p_orig = Pipeline(
            name="full_pipeline",
            description="three blocks",
            steps=[b1, b2, b3],
            params={"load": {"path": "/data/subject01.snirf"}},
        )

        d_first = p_orig.to_dict()
        p_loaded = Pipeline.from_dict(d_first, registry)
        d_second = p_loaded.to_dict()

        assert d_first == d_second

    def test_empty_pipeline_roundtrip_is_stable(self) -> None:
        p = Pipeline(name="empty", description="")
        d1 = p.to_dict()
        p2 = Pipeline.from_dict(d1, FakeRegistry({}))
        d2 = p2.to_dict()
        assert d1 == d2
