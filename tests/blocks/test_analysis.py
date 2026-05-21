"""Tests for analysis blocks -- BlockAverage (T-004)."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspy.blocks.analysis import (
    BlockAverageBlock,
    BlockAverageParams,
    ConditionWindow,
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


class TestPerConditionWindows:
    """Tests for per-condition temporal windows (T-012)."""

    def test_empty_per_condition_uses_defaults(
        self, raw_haemo_with_events, context
    ):
        """Empty per_condition_windows preserves legacy behaviour."""
        params = BlockAverageParams(per_condition_windows={})
        block = BlockAverageBlock(params=params)
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert isinstance(result.data, dict)
        assert len(result.data) == 2
        assert "per_condition_used" not in result.metadata

    def test_partial_override(self, raw_haemo_with_events, context):
        """Override one condition; other uses global defaults."""
        params = BlockAverageParams(
            tmin=-2.0,
            tmax=18.0,
            baseline_tmin=-2.0,
            baseline_tmax=0.0,
            per_condition_windows={
                "Tapping": ConditionWindow(
                    tmin=-5.0, tmax=25.0,
                    baseline_tmin=-5.0, baseline_tmax=0.0,
                ),
            },
        )
        block = BlockAverageBlock(params=params)
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert result.metadata["per_condition_used"] is True
        ws = result.metadata["windows_used"]
        assert ws["Tapping"]["tmin"] == -5.0
        assert ws["Tapping"]["tmax"] == 25.0
        # Rest uses global
        assert ws["Rest"]["tmin"] == -2.0
        assert ws["Rest"]["tmax"] == 18.0

    def test_full_override(self, raw_haemo_with_events, context):
        """Override all conditions."""
        params = BlockAverageParams(
            per_condition_windows={
                "Tapping": ConditionWindow(
                    tmin=-3.0, tmax=20.0,
                    baseline_tmin=-3.0, baseline_tmax=0.0,
                ),
                "Rest": ConditionWindow(
                    tmin=-1.0, tmax=10.0,
                    baseline_tmin=-1.0, baseline_tmax=0.0,
                ),
            },
        )
        block = BlockAverageBlock(params=params)
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert result.metadata["per_condition_used"] is True
        assert "Tapping" in result.data
        assert "Rest" in result.data

    def test_raises_on_unknown_condition(
        self, raw_haemo_with_events, context
    ):
        """Raise if per_condition_windows key not in event_id."""
        params = BlockAverageParams(
            per_condition_windows={
                "NonExistent": ConditionWindow(
                    tmin=-2.0, tmax=18.0,
                    baseline_tmin=-2.0, baseline_tmax=0.0,
                ),
            },
        )
        block = BlockAverageBlock(params=params)
        with pytest.raises(ValidationError, match="per_condition_windows"):
            block.run(context, {"beer_lambert": raw_haemo_with_events})

    def test_raises_on_bad_condition_window_tmin_tmax(self):
        """Raise if ConditionWindow has tmin >= tmax."""
        params = BlockAverageParams(
            per_condition_windows={
                "Tapping": ConditionWindow(
                    tmin=10.0, tmax=5.0,
                    baseline_tmin=-2.0, baseline_tmax=0.0,
                ),
            },
        )
        block = BlockAverageBlock(params=params)
        with pytest.raises(ValidationError, match="tmin"):
            block.run(None, {"beer_lambert": "fake"})

    def test_raises_on_bad_condition_window_baseline(self):
        """Raise if ConditionWindow has baseline_tmin > baseline_tmax."""
        params = BlockAverageParams(
            per_condition_windows={
                "Tapping": ConditionWindow(
                    tmin=-2.0, tmax=18.0,
                    baseline_tmin=1.0, baseline_tmax=-1.0,
                ),
            },
        )
        block = BlockAverageBlock(params=params)
        with pytest.raises(ValidationError, match="baseline_tmin"):
            block.run(None, {"beer_lambert": "fake"})

    def test_post_init_coerces_dicts(self):
        """__post_init__ converts raw dicts to ConditionWindow."""
        params = BlockAverageParams(
            per_condition_windows={
                "A": {"tmin": -1.0, "tmax": 10.0,
                       "baseline_tmin": -1.0, "baseline_tmax": 0.0},
            },
        )
        assert isinstance(params.per_condition_windows["A"], ConditionWindow)
        assert params.per_condition_windows["A"].tmin == -1.0

    def test_post_init_partial_dict_only_tmin(self):
        """Partial dict with only tmin fills missing fields from global defaults."""
        params = BlockAverageParams(
            tmin=-2.0,
            tmax=18.0,
            baseline_tmin=-2.0,
            baseline_tmax=0.0,
            per_condition_windows={"CondA": {"tmin": -3.0}},
        )
        window = params.per_condition_windows["CondA"]
        assert isinstance(window, ConditionWindow)
        assert window.tmin == -3.0
        assert window.tmax == 18.0
        assert window.baseline_tmin == -2.0
        assert window.baseline_tmax == 0.0

    def test_post_init_partial_dict_only_tmax(self):
        """Partial dict with only tmax fills missing fields from global defaults."""
        params = BlockAverageParams(
            tmin=-2.0,
            tmax=18.0,
            baseline_tmin=-2.0,
            baseline_tmax=0.0,
            per_condition_windows={"CondB": {"tmax": 25.0}},
        )
        window = params.per_condition_windows["CondB"]
        assert isinstance(window, ConditionWindow)
        assert window.tmin == -2.0
        assert window.tmax == 25.0
        assert window.baseline_tmin == -2.0
        assert window.baseline_tmax == 0.0

    def test_post_init_empty_dict_per_condition_uses_global_defaults(self):
        """Empty dict {} for a condition uses all global defaults."""
        params = BlockAverageParams(
            tmin=-1.5,
            tmax=20.0,
            baseline_tmin=-1.5,
            baseline_tmax=0.0,
            per_condition_windows={"CondC": {}},
        )
        window = params.per_condition_windows["CondC"]
        assert isinstance(window, ConditionWindow)
        assert window.tmin == -1.5
        assert window.tmax == 20.0
        assert window.baseline_tmin == -1.5
        assert window.baseline_tmax == 0.0

    def test_partial_dict_pipeline_does_not_crash(
        self, raw_haemo_with_events, context
    ):
        """Pipeline with partially-filled per_condition_windows runs without error."""
        params = BlockAverageParams(
            tmin=-2.0,
            tmax=18.0,
            baseline_tmin=-2.0,
            baseline_tmax=0.0,
            per_condition_windows={"Tapping": {"tmin": -3.0}},
        )
        block = BlockAverageBlock(params=params)
        result = block.run(context, {"beer_lambert": raw_haemo_with_events})
        assert result.metadata["per_condition_used"] is True
        ws = result.metadata["windows_used"]
        assert ws["Tapping"]["tmin"] == -3.0
        assert ws["Tapping"]["tmax"] == 18.0
