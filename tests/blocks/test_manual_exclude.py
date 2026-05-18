"""Tests for ManualChannelExcludeBlock (T-011 scope D)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from nirspy.blocks.manual_exclude import (
    ManualChannelExcludeBlock,
    ManualChannelExcludeParams,
)
from nirspy.domain.exceptions import ValidationError


def _make_raw(ch_names: list[str], bads: list[str] | None = None) -> Any:
    """Create a minimal mock Raw with ch_names and info["bads"]."""
    raw = MagicMock()
    raw.ch_names = ch_names
    raw.info = {"bads": list(bads or [])}
    raw.copy.return_value = MagicMock()
    raw.copy.return_value.ch_names = ch_names
    raw.copy.return_value.info = {"bads": list(bads or [])}
    return raw


class TestManualExcludeBasic:

    def test_empty_channels_passthrough(self) -> None:
        raw = _make_raw(["S1_D1 760", "S1_D1 850"])
        block = ManualChannelExcludeBlock()
        result = block.run(None, {"prev": raw})
        assert result.metadata["n_excluded"] == 0
        assert result.data is raw  # no copy when empty

    def test_marks_channel_as_bad(self) -> None:
        raw = _make_raw(["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"])
        params = ManualChannelExcludeParams(channels=["S1_D1 760"])
        block = ManualChannelExcludeBlock(params=params)
        result = block.run(None, {"prev": raw})
        bads = result.data.info["bads"]
        assert "S1_D1 760" in bads

    def test_no_input_raises(self) -> None:
        block = ManualChannelExcludeBlock()
        with pytest.raises(ValidationError):
            block.run(None, {})


class TestWavelengthPairing:

    def test_expands_to_pair(self) -> None:
        raw = _make_raw(["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"])
        params = ManualChannelExcludeParams(channels=["S1_D1 760"])
        block = ManualChannelExcludeBlock(params=params)
        result = block.run(None, {"prev": raw})
        bads = result.data.info["bads"]
        # Should expand to both wavelengths
        assert "S1_D1 760" in bads
        assert "S1_D1 850" in bads

    def test_preserves_existing_bads(self) -> None:
        raw = _make_raw(
            ["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"],
            bads=["S2_D1 760", "S2_D1 850"],
        )
        params = ManualChannelExcludeParams(channels=["S1_D1 760"])
        block = ManualChannelExcludeBlock(params=params)
        result = block.run(None, {"prev": raw})
        bads = result.data.info["bads"]
        assert "S2_D1 760" in bads
        assert "S1_D1 760" in bads


class TestInvalidChannel:

    def test_nonexistent_channel_raises(self) -> None:
        raw = _make_raw(["S1_D1 760", "S1_D1 850"])
        params = ManualChannelExcludeParams(channels=["FAKE_CHANNEL"])
        block = ManualChannelExcludeBlock(params=params)
        with pytest.raises(ValidationError, match="not found"):
            block.run(None, {"prev": raw})


class TestBlockSpec:

    def test_spec_any_any(self) -> None:
        from nirspy.domain.data_types import DataType
        block = ManualChannelExcludeBlock()
        assert block.spec.input_type == DataType.ANY
        assert block.spec.output_type == DataType.ANY

    def test_registered_in_registry(self) -> None:
        from nirspy.blocks import registry
        assert "manual_channel_exclude" in registry
