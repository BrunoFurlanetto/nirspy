"""Tests for BlockSpec, BlockResult and Block Protocol invariants."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from nirspy.domain.block import Block, BlockResult, BlockSpec
from nirspy.domain.data_types import DataType


class TestBlockSpec:
    """BlockSpec is a frozen dataclass — immutable and hashable."""

    def test_minimal_construction(self) -> None:
        spec = BlockSpec(
            block_id="load",
            display_name="Load SNIRF",
            input_type=DataType.ANY,
            output_type=DataType.RAW,
        )
        assert spec.block_id == "load"
        assert spec.output_type is DataType.RAW

    def test_defaults(self) -> None:
        spec = BlockSpec(
            block_id="x",
            display_name="X",
            input_type=DataType.RAW,
            output_type=DataType.RAW_OD,
        )
        assert spec.enabled is True
        assert spec.params_class is None
        assert spec.description == ""

    def test_immutability(self) -> None:
        spec = BlockSpec(
            block_id="load",
            display_name="Load SNIRF",
            input_type=DataType.ANY,
            output_type=DataType.RAW,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            spec.block_id = "other"  # type: ignore[misc]

    def test_hashable(self) -> None:
        spec = BlockSpec(
            block_id="a",
            display_name="A",
            input_type=DataType.RAW,
            output_type=DataType.RAW_OD,
        )
        s = {spec}
        assert spec in s

    def test_equality(self) -> None:
        spec_a = BlockSpec("a", "A", DataType.RAW, DataType.RAW_OD)
        spec_b = BlockSpec("a", "A", DataType.RAW, DataType.RAW_OD)
        assert spec_a == spec_b

    def test_inequality_different_ids(self) -> None:
        spec_a = BlockSpec("a", "A", DataType.RAW, DataType.RAW_OD)
        spec_b = BlockSpec("b", "A", DataType.RAW, DataType.RAW_OD)
        assert spec_a != spec_b

    def test_enabled_false(self) -> None:
        spec = BlockSpec(
            block_id="disabled_block",
            display_name="Disabled",
            input_type=DataType.RAW,
            output_type=DataType.RAW_OD,
            enabled=False,
        )
        assert spec.enabled is False

    def test_params_class_reference(self) -> None:
        @dataclasses.dataclass
        class MyParams:
            threshold: float = 0.5

        spec = BlockSpec(
            block_id="with_params",
            display_name="With Params",
            input_type=DataType.RAW,
            output_type=DataType.RAW_OD,
            params_class=MyParams,
        )
        assert spec.params_class is MyParams


class TestBlockResult:
    """BlockResult is a plain mutable dataclass."""

    def test_construction(self) -> None:
        result = BlockResult(data="some_data", block_id="my_block")
        assert result.data == "some_data"
        assert result.block_id == "my_block"
        assert result.metadata == {}

    def test_metadata_populated(self) -> None:
        result = BlockResult(data=42, block_id="b1", metadata={"duration_s": 1.2})
        assert result.metadata["duration_s"] == pytest.approx(1.2)

    def test_metadata_default_is_independent(self) -> None:
        r1 = BlockResult(data=1, block_id="a")
        r2 = BlockResult(data=2, block_id="b")
        r1.metadata["x"] = 1
        assert "x" not in r2.metadata

    def test_data_can_be_any_type(self) -> None:
        for payload in [None, [], {}, object(), b"bytes"]:
            result = BlockResult(data=payload, block_id="x")
            assert result.data is payload


class TestBlockProtocol:
    """Block is a runtime_checkable Protocol."""

    def test_fake_block_satisfies_protocol(self, fake_block_raw_to_raw: Block) -> None:
        assert isinstance(fake_block_raw_to_raw, Block)

    def test_spec_accessible(self, fake_block_raw_to_raw: Block) -> None:
        assert isinstance(fake_block_raw_to_raw.spec, BlockSpec)

    def test_run_returns_block_result(
        self, fake_block_raw_to_raw: Block, default_context: Any
    ) -> None:
        # New signature: run(context, inputs) — ADR-009
        result = fake_block_raw_to_raw.run(default_context, {})
        assert isinstance(result, BlockResult)

    def test_run_passes_data_through(
        self, fake_block_raw_to_raw: Block, default_context: Any
    ) -> None:
        # When inputs carries one upstream value, FakeBlock passes it through.
        sentinel = object()
        result = fake_block_raw_to_raw.run(
            default_context, {"upstream": sentinel}
        )
        assert result.data is sentinel

    def test_run_first_block_receives_empty_inputs(
        self, fake_block_raw_to_raw: Block, default_context: Any
    ) -> None:
        # First block in a pipeline always receives inputs={}.
        result = fake_block_raw_to_raw.run(default_context, {})
        assert result.block_id == fake_block_raw_to_raw.spec.block_id

    def test_params_accessible_via_self(self, default_context: Any) -> None:
        # ADR-009: blocks receive params at construction time, not via run().
        import dataclasses as _dc

        @_dc.dataclass
        class _P:
            threshold: float = 0.5

        from tests.conftest import make_block as _mb
        from nirspy.domain.data_types import DataType as _DT

        block = _mb("with_params", _DT.RAW, _DT.RAW, params=_P(threshold=0.9))
        assert block.params.threshold == 0.9  # type: ignore[attr-defined]
