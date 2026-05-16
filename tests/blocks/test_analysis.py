"""Tests for analysis blocks -- BlockAverage (T-004)."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspy.blocks.analysis import (
    BlockAverageBlock,
    BlockAverageParams,
)
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.execution import ExecutionContext


@pytest.fixture()
def raw_haemo_with_events() -> mne.io.BaseRaw:
    """Raw haemo data with stimulus annotations for epoching."""
    sfreq = 10.0
    n_times = int(60 * sfreq)  # 60 seconds
    n_channels = 4

    ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"]
    ch_types = ["hbo", "hbr", "hbo", "hbr"]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Set source/detector positions
    sources = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
    detectors = np.array([[0.015, 0.0, 0.0]])
    for i, ch in enumerate(info["chs"]):
        src_idx = i // 2
        ch["loc"][3:6] = sources[src_idx]
        ch["loc"][6:9] = detectors[0]

    # Generate synthetic HRF-like data
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1e-6, (n_channels, n_times))
    raw = mne.io.RawArray(data, info, verbose=False)

    # Add stimulus annotations (2 conditions, 3 events each)
    onsets = [5.0, 15.0, 25.0, 10.0, 20.0, 30.0]
    durations = [0.0] * 6
    descriptions = ["Tapping", "Tapping", "Tapping", "Rest", "Rest", "Rest"]
    annotations = mne.Annotations(onsets, durations, descriptions)
    raw.set_annotations(annotations)

    return raw


@pytest.fixture()
def raw_haemo_no_events() -> mne.io.BaseRaw:
    """Raw haemo data WITHOUT annotations."""
    sfreq = 10.0
    n_times = int(30 * sfreq)
    ch_names = ["S1_D1 hbo", "S1_D1 hbr"]
    ch_types = ["hbo", "hbr"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    sources = np.array([[0.0, 0.0, 0.0]])
    detectors = np.array([[0.015, 0.0, 0.0]])
    for _i, ch in enumerate(info["chs"]):
        ch["loc"][3:6] = sources[0]
        ch["loc"][6:9] = detectors[0]
    rng = np.random.default_rng(99)
    data = rng.normal(0, 1e-6, (2, n_times))
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture()
def context() -> ExecutionContext:
    """Fresh execution context."""
    return ExecutionContext()


class TestBlockAverageBlock:
    """Tests for BlockAverageBlock."""

    def test_returns_evoked_dict(self, raw_haemo_with_events, context):
        block = BlockAverageBlock()
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert isinstance(result.data, dict)
        for key, val in result.data.items():
            assert isinstance(key, str)
            assert isinstance(val, mne.Evoked)

    def test_one_evoked_per_condition(self, raw_haemo_with_events, context):
        block = BlockAverageBlock()
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert "Tapping" in result.data
        assert "Rest" in result.data
        assert len(result.data) == 2

    def test_metadata_has_conditions(self, raw_haemo_with_events, context):
        block = BlockAverageBlock()
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert "conditions" in result.metadata
        assert "n_conditions" in result.metadata
        assert result.metadata["n_conditions"] == 2

    def test_block_id(self, raw_haemo_with_events, context):
        block = BlockAverageBlock()
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert result.block_id == "block_average"

    def test_raises_on_no_events(self, raw_haemo_no_events, context):
        block = BlockAverageBlock()
        with pytest.raises(ValidationError, match="no events found"):
            block.run(context, {"beer_lambert": raw_haemo_no_events})

    def test_raises_on_empty_inputs(self, context):
        block = BlockAverageBlock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(context, {})

    def test_raises_on_wrong_ch_type(self, context):
        info = mne.create_info(ch_names=["S1_D1 760"], sfreq=10.0,
                               ch_types=["fnirs_od"])
        for ch in info["chs"]:
            ch["loc"][3:6] = [0.0, 0.0, 0.0]
            ch["loc"][6:9] = [0.015, 0.0, 0.0]
            ch["loc"][9] = 760.0
        data = np.ones((1, 100))
        raw = mne.io.RawArray(data, info, verbose=False)
        block = BlockAverageBlock()
        with pytest.raises(ValidationError, match="hbo/hbr"):
            block.run(context, {"bandpass_filter": raw})

    def test_raises_on_bad_baseline(self, raw_haemo_with_events, context):
        params = BlockAverageParams(baseline_tmin=1.0, baseline_tmax=-1.0)
        block = BlockAverageBlock(params=params)
        with pytest.raises(ValidationError, match="baseline_tmin"):
            block.run(context, {"beer_lambert": raw_haemo_with_events})

    def test_raises_on_bad_tmin_tmax(self, raw_haemo_with_events, context):
        params = BlockAverageParams(tmin=10.0, tmax=5.0)
        block = BlockAverageBlock(params=params)
        with pytest.raises(ValidationError, match="tmin"):
            block.run(context, {"beer_lambert": raw_haemo_with_events})

    def test_pick_conditions(self, raw_haemo_with_events, context):
        params = BlockAverageParams(pick_conditions=["Tapping"])
        block = BlockAverageBlock(params=params)
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert "Tapping" in result.data
        assert "Rest" not in result.data

    def test_pick_nonexistent_condition(self, raw_haemo_with_events, context):
        params = BlockAverageParams(pick_conditions=["NonExistent"])
        block = BlockAverageBlock(params=params)
        with pytest.raises(ValidationError, match="pick_conditions"):
            block.run(context, {"beer_lambert": raw_haemo_with_events})

    def test_spec(self):
        block = BlockAverageBlock()
        assert block.spec.block_id == "block_average"
        assert block.spec.params_class is BlockAverageParams

    def test_custom_epoch_window(self, raw_haemo_with_events, context):
        params = BlockAverageParams(tmin=-1.0, tmax=10.0, baseline_tmin=-1.0, baseline_tmax=0.0)
        block = BlockAverageBlock(params=params)
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert isinstance(result.data, dict)
        assert len(result.data) > 0
