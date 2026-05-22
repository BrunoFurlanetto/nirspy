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


class TestSplineBlock:
    """Tests for SplineBlock (T-016)."""

    def test_spline_output_type(self) -> None:
        """Spline output type is RAW_OD (OD in -> OD out)."""
        from nirspy.blocks.motion import SplineBlock

        block = SplineBlock()
        assert block.spec.input_type == DataType.RAW_OD
        assert block.spec.output_type == DataType.RAW_OD

    def test_spline_preserves_shape(self, raw_od: mne.io.BaseRaw) -> None:
        """Spline preserves n_channels and n_times."""
        from nirspy.blocks.motion import SplineBlock

        block = SplineBlock()
        ctx = MagicMock()
        result = block.run(ctx, {"prev": raw_od})
        assert isinstance(result, BlockResult)
        out_raw: mne.io.BaseRaw = result.data
        assert out_raw.info["nchan"] == raw_od.info["nchan"]
        assert out_raw.n_times == raw_od.n_times

    def test_spline_custom_params(self, raw_od: mne.io.BaseRaw) -> None:
        """Spline respects custom threshold and spline_order."""
        from nirspy.blocks.motion import SplineBlock, SplineParams

        params = SplineParams(threshold=5.0, spline_order=2)
        block = SplineBlock(params=params)
        assert block.params.threshold == 5.0
        assert block.params.spline_order == 2

        ctx = MagicMock()
        result = block.run(ctx, {"prev": raw_od})
        assert isinstance(result, BlockResult)
        assert result.metadata["threshold"] == 5.0
        assert result.metadata["spline_order"] == 2

    def test_spline_spec_registration(self) -> None:
        """Spline block is registered in the block registry."""
        from nirspy.blocks import registry
        from nirspy.blocks.motion import SplineBlock

        block_cls = registry.get("spline_motion_correction")
        assert block_cls is SplineBlock

    def test_spline_requires_input(self) -> None:
        """Spline raises ValidationError when inputs are empty."""
        from nirspy.blocks.motion import SplineBlock

        block = SplineBlock()
        ctx = MagicMock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(ctx, {})

    def test_spline_rejects_wrong_channel_type(
        self, raw_cw_amplitude: mne.io.BaseRaw
    ) -> None:
        """Spline raises ValidationError when input is not OD."""
        from nirspy.blocks.motion import SplineBlock

        block = SplineBlock()
        ctx = MagicMock()
        with pytest.raises(ValidationError, match="fnirs_od"):
            block.run(ctx, {"prev": raw_cw_amplitude})

    def test_spline_params_dataclass(self) -> None:
        """SplineParams is a frozen dataclass with threshold + spline_order."""
        import dataclasses

        from nirspy.blocks.motion import SplineParams

        assert dataclasses.is_dataclass(SplineParams)
        params = SplineParams()
        fields = {f.name for f in dataclasses.fields(params)}
        assert fields == {"threshold", "spline_order"}
        assert params.threshold == 3.0
        assert params.spline_order == 3

    def test_spline_metadata(self, raw_od: mne.io.BaseRaw) -> None:
        """Spline result metadata includes method and reference."""
        from nirspy.blocks.motion import SplineBlock

        block = SplineBlock()
        ctx = MagicMock()
        result = block.run(ctx, {"prev": raw_od})
        assert result.metadata["method"] == "spline"
        assert "Scholkmann" in result.metadata["reference"]

    def test_spline_block_id(self) -> None:
        """Spline block_id is 'spline_motion_correction'."""
        from nirspy.blocks.motion import SplineBlock

        block = SplineBlock()
        assert block.spec.block_id == "spline_motion_correction"
        assert block.spec.display_name == "Spline Motion Correction"


class TestWaveletBlock:
    """Tests for WaveletBlock (T-017)."""

    def test_wavelet_output_type(self) -> None:
        """Wavelet output type is RAW_OD (OD in -> OD out)."""
        from nirspy.blocks.motion import WaveletBlock

        block = WaveletBlock()
        assert block.spec.input_type == DataType.RAW_OD
        assert block.spec.output_type == DataType.RAW_OD

    def test_wavelet_preserves_shape(self, raw_od: mne.io.BaseRaw) -> None:
        """Wavelet preserves n_channels and n_times."""
        from nirspy.blocks.motion import WaveletBlock

        block = WaveletBlock()
        ctx = MagicMock()
        result = block.run(ctx, {"prev": raw_od})
        assert isinstance(result, BlockResult)
        out_raw: mne.io.BaseRaw = result.data
        assert out_raw.info["nchan"] == raw_od.info["nchan"]
        assert out_raw.n_times == raw_od.n_times

    def test_wavelet_custom_params(self, raw_od: mne.io.BaseRaw) -> None:
        """Wavelet respects custom wavelet and iqr_multiplier."""
        from nirspy.blocks.motion import WaveletBlock, WaveletParams

        params = WaveletParams(wavelet="db4", iqr_multiplier=2.0)
        block = WaveletBlock(params=params)
        assert block.params.wavelet == "db4"
        assert block.params.iqr_multiplier == 2.0

        ctx = MagicMock()
        result = block.run(ctx, {"prev": raw_od})
        assert isinstance(result, BlockResult)
        assert result.metadata["wavelet"] == "db4"
        assert result.metadata["iqr_multiplier"] == 2.0

    def test_wavelet_spec_registration(self) -> None:
        """Wavelet block is registered in the block registry."""
        from nirspy.blocks import registry
        from nirspy.blocks.motion import WaveletBlock

        block_cls = registry.get("wavelet_motion_correction")
        assert block_cls is WaveletBlock

    def test_wavelet_requires_input(self) -> None:
        """Wavelet raises ValidationError when inputs are empty."""
        from nirspy.blocks.motion import WaveletBlock

        block = WaveletBlock()
        ctx = MagicMock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(ctx, {})

    def test_wavelet_rejects_wrong_channel_type(
        self, raw_cw_amplitude: mne.io.BaseRaw
    ) -> None:
        """Wavelet raises ValidationError when input is not OD."""
        from nirspy.blocks.motion import WaveletBlock

        block = WaveletBlock()
        ctx = MagicMock()
        with pytest.raises(ValidationError, match="fnirs_od"):
            block.run(ctx, {"prev": raw_cw_amplitude})

    def test_wavelet_params_dataclass(self) -> None:
        """WaveletParams is a frozen dataclass with wavelet + iqr_multiplier."""
        import dataclasses

        from nirspy.blocks.motion import WaveletParams

        assert dataclasses.is_dataclass(WaveletParams)
        params = WaveletParams()
        fields = {f.name for f in dataclasses.fields(params)}
        assert fields == {"wavelet", "iqr_multiplier"}
        assert params.wavelet == "sym8"
        assert params.iqr_multiplier == 1.5

    def test_wavelet_metadata(self, raw_od: mne.io.BaseRaw) -> None:
        """Wavelet result metadata includes method and reference."""
        from nirspy.blocks.motion import WaveletBlock

        block = WaveletBlock()
        ctx = MagicMock()
        result = block.run(ctx, {"prev": raw_od})
        assert result.metadata["method"] == "wavelet"
        assert "Molavi" in result.metadata["reference"]

    def test_wavelet_block_id(self) -> None:
        """Wavelet block_id is 'wavelet_motion_correction'."""
        from nirspy.blocks.motion import WaveletBlock

        block = WaveletBlock()
        assert block.spec.block_id == "wavelet_motion_correction"
        assert block.spec.display_name == "Wavelet Motion Correction"
