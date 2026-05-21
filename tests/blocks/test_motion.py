"""Tests for motion correction blocks (T-015)."""

from __future__ import annotations

from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from nirspy.blocks.motion import TDDRBlock, TDDRParams
from nirspy.domain.block import BlockResult
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError


class TestTDDRBlock:
    """Tests for TDDRBlock."""

    def test_tddr_output_type(self, raw_od: mne.io.BaseRaw) -> None:
        """TDDR output type is RAW_OD (OD in -> OD out)."""
        block = TDDRBlock()
        assert block.spec.input_type == DataType.RAW_OD
        assert block.spec.output_type == DataType.RAW_OD

    def test_tddr_preserves_shape(self, raw_od: mne.io.BaseRaw) -> None:
        """TDDR preserves n_channels and n_times."""
        block = TDDRBlock()
        ctx = MagicMock()
        result = block.run(ctx, {"prev": raw_od})
        assert isinstance(result, BlockResult)
        out_raw: mne.io.BaseRaw = result.data
        assert out_raw.info["nchan"] == raw_od.info["nchan"]
        assert out_raw.n_times == raw_od.n_times

    def test_tddr_modifies_data(self, raw_od: mne.io.BaseRaw) -> None:
        """TDDR modifies signal data (output != input for noisy signal)."""
        # Inject a spike artifact to ensure TDDR modifies data
        raw_copy = raw_od.copy()
        data = raw_copy.get_data()
        data[0, 50] += 10.0  # large spike
        raw_copy._data = data  # type: ignore[attr-defined]

        block = TDDRBlock()
        ctx = MagicMock()
        result = block.run(ctx, {"prev": raw_copy})
        out_data = result.data.get_data()
        assert not np.array_equal(data, out_data)

    def test_tddr_spec_registration(self) -> None:
        """TDDR block is registered in the block registry."""
        from nirspy.blocks import registry

        block_cls = registry.get("tddr")
        assert block_cls is TDDRBlock

    def test_tddr_requires_input(self) -> None:
        """TDDR raises ValidationError when inputs are empty."""
        block = TDDRBlock()
        ctx = MagicMock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(ctx, {})

    def test_tddr_rejects_wrong_channel_type(
        self, raw_cw_amplitude: mne.io.BaseRaw
    ) -> None:
        """TDDR raises ValidationError when input is not OD."""
        block = TDDRBlock()
        ctx = MagicMock()
        with pytest.raises(ValidationError, match="fnirs_od"):
            block.run(ctx, {"prev": raw_cw_amplitude})

    def test_tddr_params_dataclass(self) -> None:
        """TDDRParams is a frozen dataclass (registry consistency)."""
        import dataclasses

        assert dataclasses.is_dataclass(TDDRParams)
        params = TDDRParams()
        assert len(dataclasses.fields(params)) == 0

    def test_tddr_metadata(self, raw_od: mne.io.BaseRaw) -> None:
        """TDDR result metadata includes method and reference."""
        block = TDDRBlock()
        ctx = MagicMock()
        result = block.run(ctx, {"prev": raw_od})
        assert result.metadata["method"] == "tddr"
        assert "Fishburn" in result.metadata["reference"]

    def test_tddr_block_id(self) -> None:
        """TDDR block_id is 'tddr'."""
        block = TDDRBlock()
        assert block.spec.block_id == "tddr"
        assert block.spec.display_name == "TDDR Motion Correction"
