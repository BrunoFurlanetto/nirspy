"""Tests for preprocessing blocks (T-003)."""

from __future__ import annotations

import mne
import pytest

from nirspy.blocks.preprocessing import (
    BandpassFilterBlock,
    BandpassFilterParams,
    BeerLambertBlock,
    BeerLambertParams,
    OpticalDensityBlock,
)
from nirspy.domain.exceptions import ValidationError


class TestOpticalDensityBlock:
    """Tests for OpticalDensityBlock."""

    def test_converts_cw_amplitude_to_od(self, raw_cw_amplitude: mne.io.BaseRaw) -> None:
        block = OpticalDensityBlock()
        result = block.run(None, {"load_snirf": raw_cw_amplitude})
        assert result.block_id == "optical_density"
        ch_types = set(result.data.get_channel_types())
        assert "fnirs_od" in ch_types
        assert "fnirs_cw_amplitude" not in ch_types

    def test_raises_on_empty_inputs(self) -> None:
        block = OpticalDensityBlock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(None, {})

    def test_raises_on_wrong_channel_type(self, raw_od: mne.io.BaseRaw) -> None:
        block = OpticalDensityBlock()
        with pytest.raises(ValidationError, match="fnirs_cw_amplitude"):
            block.run(None, {"prev": raw_od})

    def test_metadata_has_n_channels(self, raw_cw_amplitude: mne.io.BaseRaw) -> None:
        block = OpticalDensityBlock()
        result = block.run(None, {"load_snirf": raw_cw_amplitude})
        assert result.metadata["n_channels"] > 0

class TestBeerLambertBlock:
    """Tests for BeerLambertBlock."""

    def test_converts_od_to_haemo(self, raw_od: mne.io.BaseRaw) -> None:
        block = BeerLambertBlock()
        result = block.run(None, {"optical_density": raw_od})
        assert result.block_id == "beer_lambert"
        ch_types = set(result.data.get_channel_types())
        assert "hbo" in ch_types
        assert "hbr" in ch_types

    def test_custom_ppf(self, raw_od: mne.io.BaseRaw) -> None:
        block = BeerLambertBlock(params=BeerLambertParams(ppf=5.0))
        result = block.run(None, {"optical_density": raw_od})
        assert result.metadata["ppf"] == 5.0

    def test_raises_on_empty_inputs(self) -> None:
        block = BeerLambertBlock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(None, {})

    def test_raises_on_wrong_channel_type(self, raw_cw_amplitude: mne.io.BaseRaw) -> None:
        block = BeerLambertBlock()
        with pytest.raises(ValidationError, match="fnirs_od"):
            block.run(None, {"prev": raw_cw_amplitude})

    def test_raises_on_zero_ppf(self, raw_od: mne.io.BaseRaw) -> None:
        block = BeerLambertBlock(params=BeerLambertParams(ppf=0.0))
        with pytest.raises(ValidationError, match="ppf must be > 0"):
            block.run(None, {"prev": raw_od})

    def test_raises_on_negative_ppf(self, raw_od: mne.io.BaseRaw) -> None:
        block = BeerLambertBlock(params=BeerLambertParams(ppf=-1.0))
        with pytest.raises(ValidationError, match="ppf must be > 0"):
            block.run(None, {"prev": raw_od})

class TestBandpassFilterBlock:
    """Tests for BandpassFilterBlock."""

    def test_filters_od_data(self, raw_od: mne.io.BaseRaw) -> None:
        block = BandpassFilterBlock()
        result = block.run(None, {"optical_density": raw_od})
        assert result.block_id == "bandpass_filter"
        assert result.data.get_data().shape == raw_od.get_data().shape

    def test_filters_haemo_data(self, raw_haemo: mne.io.BaseRaw) -> None:
        block = BandpassFilterBlock()
        result = block.run(None, {"beer_lambert": raw_haemo})
        assert result.data.get_data().shape == raw_haemo.get_data().shape

    def test_custom_freq_params(self, raw_od: mne.io.BaseRaw) -> None:
        params = BandpassFilterParams(l_freq=0.02, h_freq=0.3)
        block = BandpassFilterBlock(params=params)
        result = block.run(None, {"prev": raw_od})
        assert result.metadata["l_freq"] == 0.02
        assert result.metadata["h_freq"] == 0.3

    def test_highpass_only(self, raw_od: mne.io.BaseRaw) -> None:
        params = BandpassFilterParams(l_freq=0.01, h_freq=None)
        block = BandpassFilterBlock(params=params)
        result = block.run(None, {"prev": raw_od})
        assert result.metadata["h_freq"] is None

    def test_lowpass_only(self, raw_od: mne.io.BaseRaw) -> None:
        params = BandpassFilterParams(l_freq=None, h_freq=0.5)
        block = BandpassFilterBlock(params=params)
        result = block.run(None, {"prev": raw_od})
        assert result.metadata["l_freq"] is None

    def test_raises_on_empty_inputs(self) -> None:
        block = BandpassFilterBlock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(None, {})

    def test_raises_on_both_none(self, raw_od: mne.io.BaseRaw) -> None:
        params = BandpassFilterParams(l_freq=None, h_freq=None)
        block = BandpassFilterBlock(params=params)
        with pytest.raises(ValidationError, match="at least one"):
            block.run(None, {"prev": raw_od})

    def test_raises_on_l_freq_ge_h_freq(self, raw_od: mne.io.BaseRaw) -> None:
        params = BandpassFilterParams(l_freq=0.5, h_freq=0.01)
        block = BandpassFilterBlock(params=params)
        with pytest.raises(ValidationError, match="l_freq"):
            block.run(None, {"prev": raw_od})

    def test_raises_on_equal_freqs(self, raw_od: mne.io.BaseRaw) -> None:
        params = BandpassFilterParams(l_freq=0.1, h_freq=0.1)
        block = BandpassFilterBlock(params=params)
        with pytest.raises(ValidationError, match="l_freq"):
            block.run(None, {"prev": raw_od})

class TestPreprocessingChain:
    """Integration tests for the full preprocessing pipeline."""

    def test_load_od_bandpass_mbll(self, raw_cw_amplitude: mne.io.BaseRaw) -> None:
        """Full chain: LoadSnirf > OD > Bandpass > BeerLambert."""
        od_block = OpticalDensityBlock()
        od_result = od_block.run(None, {"load_snirf": raw_cw_amplitude})
        bp_block = BandpassFilterBlock()
        bp_result = bp_block.run(None, {"optical_density": od_result.data})
        bl_block = BeerLambertBlock()
        bl_result = bl_block.run(None, {"bandpass_filter": bp_result.data})
        ch_types = set(bl_result.data.get_channel_types())
        assert "hbo" in ch_types
        assert "hbr" in ch_types

    def test_load_od_mbll_without_bandpass(self, raw_cw_amplitude: mne.io.BaseRaw) -> None:
        """Chain without bandpass: LoadSnirf > OD > BeerLambert."""
        od_block = OpticalDensityBlock()
        od_result = od_block.run(None, {"load_snirf": raw_cw_amplitude})
        bl_block = BeerLambertBlock()
        bl_result = bl_block.run(None, {"optical_density": od_result.data})
        ch_types = set(bl_result.data.get_channel_types())
        assert "hbo" in ch_types


class TestRegistryIntegration:
    """Verify blocks are registered correctly."""

    def test_all_blocks_registered(self) -> None:
        from nirspy.blocks import registry
        blocks = registry.list_blocks()
        assert "optical_density" in blocks
        assert "beer_lambert" in blocks
        assert "bandpass_filter" in blocks

    def test_registry_returns_classes(self) -> None:
        from nirspy.blocks import registry
        cls = registry.get("optical_density")
        assert isinstance(cls, type)

    def test_spec_accessible_from_class(self) -> None:
        from nirspy.blocks import registry
        cls = registry.get("beer_lambert")
        assert cls.SPEC.block_id == "beer_lambert"
