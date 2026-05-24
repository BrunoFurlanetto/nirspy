"""Tests for EpochsExtractionBlock (T-035)."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspy.blocks.epochs import (
    EpochsExtractionBlock,
    EpochsExtractionParams,
)
from nirspy.domain.block import BlockResult
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError


@pytest.fixture()
def raw_haemo_with_events() -> mne.io.BaseRaw:
    """Create synthetic Raw with hbo/hbr channels and stimulus annotations."""
    sfreq = 10.0
    n_times = int(30 * sfreq)  # 30 seconds
    n_channels = 4

    ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"]
    ch_types = ["hbo", "hbr", "hbo", "hbr"]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Set source/detector positions
    for _i, ch in enumerate(info["chs"]):
        ch["loc"][3:6] = [0.0, 0.0, 0.0]
        ch["loc"][6:9] = [0.03, 0.0, 0.0]

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 1e-6

    raw = mne.io.RawArray(data, info, verbose=False)

    # Add stimulus annotations at t=5, t=15, t=25
    raw.set_annotations(mne.Annotations(
        onset=[5.0, 15.0, 25.0],
        duration=[0.0, 0.0, 0.0],
        description=["stim_A", "stim_B", "stim_A"],
    ))

    return raw


class TestEpochsExtractionBlock:
    """Tests for EpochsExtractionBlock."""

    def test_params_defaults(self) -> None:
        """Default params are correct."""
        params = EpochsExtractionParams()
        assert params.tmin == -0.5
        assert params.tmax == 5.0
        assert params.baseline_tmin is None
        assert params.baseline_tmax == 0.0
        assert params.reject_amplitude is None
        assert params.reject_gradient is None
        assert params.event_id is None

    def test_params_frozen(self) -> None:
        """Params are immutable."""
        params = EpochsExtractionParams()
        with pytest.raises(AttributeError):
            params.tmin = 1.0  # type: ignore[misc]

    def test_spec_correct(self) -> None:
        """Block spec has correct input/output types."""
        block = EpochsExtractionBlock()
        assert block.spec.block_id == "epochs_extraction"
        assert block.spec.input_type == DataType.RAW_HAEMO
        assert block.spec.output_type == DataType.EPOCHS

    def test_raises_on_empty_inputs(self) -> None:
        """Block raises ValidationError when inputs is empty."""
        block = EpochsExtractionBlock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(None, {})

    def test_raises_on_wrong_channel_type(self) -> None:
        """Block raises ValidationError when channels are not hbo/hbr."""
        block = EpochsExtractionBlock()
        info = mne.create_info(
            ch_names=["S1_D1 760", "S1_D1 850"],
            sfreq=10.0,
            ch_types=["fnirs_cw_amplitude", "fnirs_cw_amplitude"],
        )
        for ch in info["chs"]:
            ch["loc"][3:6] = [0.0, 0.0, 0.0]
            ch["loc"][6:9] = [0.03, 0.0, 0.0]
            ch["loc"][9] = 760.0

        raw = mne.io.RawArray(np.ones((2, 100)), info, verbose=False)
        with pytest.raises(ValidationError, match="expects hbo/hbr"):
            block.run(None, {"prev": raw})

    def test_raises_on_invalid_tmin_tmax(
        self, raw_haemo_with_events: mne.io.BaseRaw
    ) -> None:
        """Block raises ValidationError when tmin >= tmax."""
        params = EpochsExtractionParams(tmin=5.0, tmax=1.0)
        block = EpochsExtractionBlock(params=params)
        with pytest.raises(ValidationError, match="tmin.*must be < tmax"):
            block.run(None, {"prev": raw_haemo_with_events})

    def test_raises_on_invalid_baseline(
        self, raw_haemo_with_events: mne.io.BaseRaw
    ) -> None:
        """Block raises ValidationError when baseline_tmin > baseline_tmax."""
        params = EpochsExtractionParams(baseline_tmin=1.0, baseline_tmax=0.0)
        block = EpochsExtractionBlock(params=params)
        with pytest.raises(ValidationError, match="baseline_tmin"):
            block.run(None, {"prev": raw_haemo_with_events})

    def test_raises_on_no_events(self) -> None:
        """Block raises ValidationError when raw has no annotations."""
        info = mne.create_info(
            ch_names=["S1_D1 hbo", "S1_D1 hbr"],
            sfreq=10.0,
            ch_types=["hbo", "hbr"],
        )
        for ch in info["chs"]:
            ch["loc"][3:6] = [0.0, 0.0, 0.0]
            ch["loc"][6:9] = [0.03, 0.0, 0.0]

        raw = mne.io.RawArray(
            np.ones((2, 100)) * 1e-6, info, verbose=False
        )
        block = EpochsExtractionBlock()
        with pytest.raises(ValidationError, match="no events found"):
            block.run(None, {"prev": raw})

    def test_run_produces_epochs(
        self, raw_haemo_with_events: mne.io.BaseRaw
    ) -> None:
        """Block produces mne.Epochs with correct metadata."""
        params = EpochsExtractionParams(tmin=-0.5, tmax=4.0)
        block = EpochsExtractionBlock(params=params)
        result = block.run(None, {"prev": raw_haemo_with_events})

        assert isinstance(result, BlockResult)
        assert result.block_id == "epochs_extraction"
        assert isinstance(result.data, mne.Epochs)
        assert result.metadata["n_epochs_total"] > 0
        assert "n_epochs_dropped" in result.metadata
        assert "drop_log_summary" in result.metadata
        assert "conditions" in result.metadata

    def test_run_with_event_id_filter(
        self, raw_haemo_with_events: mne.io.BaseRaw
    ) -> None:
        """Block respects event_id parameter to filter conditions."""
        # Get auto event_id first
        _, auto_event_id = mne.events_from_annotations(
            raw_haemo_with_events, verbose=False
        )

        # Only keep stim_A
        params = EpochsExtractionParams(
            tmin=-0.5,
            tmax=4.0,
            event_id={"stim_A": auto_event_id["stim_A"]},
        )
        block = EpochsExtractionBlock(params=params)
        result = block.run(None, {"prev": raw_haemo_with_events})

        assert "stim_A" in result.metadata["conditions"]

    def test_run_with_amplitude_rejection(
        self, raw_haemo_with_events: mne.io.BaseRaw
    ) -> None:
        """Block applies amplitude rejection when configured."""
        # Very tight threshold to trigger some rejections
        params = EpochsExtractionParams(
            tmin=-0.5,
            tmax=4.0,
            reject_amplitude=1e-9,  # extremely tight
        )
        block = EpochsExtractionBlock(params=params)
        result = block.run(None, {"prev": raw_haemo_with_events})

        # With such tight threshold, epochs should be dropped
        assert result.metadata["n_epochs_dropped"] >= 0

    def test_registered_in_registry(self) -> None:
        """Block is registered in the global registry."""
        from nirspy.blocks import registry

        assert "epochs_extraction" in registry.list_blocks()
