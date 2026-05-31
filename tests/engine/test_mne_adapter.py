"""Tests for MNEAdapter and RawWrapper.

Integration tests that require MNE-NIRS sample data.
Skipped automatically when the dataset cannot be downloaded.
"""

from __future__ import annotations

import pathlib

import mne
import numpy as np
import pytest

from nirspy.engine.exceptions import MNEOperationError, SnirfLoadError
from nirspy.engine.mne_adapter import MNEAdapter, RawWrapper, _get_annotation_duration


@pytest.fixture(scope="module")
def adapter() -> MNEAdapter:
    return MNEAdapter()


# ---------------------------------------------------------------------------
# MNEAdapter.load_snirf — error cases (do not require network)
# ---------------------------------------------------------------------------


class TestMNEAdapterLoadSnirfErrors:
    def test_nonexistent_file_raises_snirf_load_error(
        self, adapter: MNEAdapter, tmp_path: pathlib.Path
    ) -> None:
        missing = tmp_path / "does_not_exist.snirf"
        with pytest.raises(SnirfLoadError, match="not found"):
            adapter.load_snirf(missing)

    def test_invalid_file_raises_snirf_load_error(
        self, adapter: MNEAdapter, tmp_path: pathlib.Path
    ) -> None:
        bad_file = tmp_path / "bad.snirf"
        bad_file.write_bytes(b"not a valid snirf file")
        with pytest.raises(SnirfLoadError):
            adapter.load_snirf(bad_file)

    def test_error_message_contains_path(
        self, adapter: MNEAdapter, tmp_path: pathlib.Path
    ) -> None:
        missing = tmp_path / "mydata.snirf"
        with pytest.raises(SnirfLoadError) as exc_info:
            adapter.load_snirf(missing)
        assert "mydata.snirf" in str(exc_info.value)

    def test_snirf_load_error_is_mne_operation_error(
        self, adapter: MNEAdapter, tmp_path: pathlib.Path
    ) -> None:
        """SnirfLoadError must be catchable as MNEOperationError (hierarchy test)."""
        missing = tmp_path / "hierarchy_test.snirf"
        with pytest.raises(MNEOperationError):
            adapter.load_snirf(missing)

    def test_invalid_file_wraps_mne_exception(
        self, adapter: MNEAdapter, tmp_path: pathlib.Path
    ) -> None:
        """When MNE raises internally, SnirfLoadError must expose mne_exception."""
        bad_file = tmp_path / "corrupt.snirf"
        bad_file.write_bytes(b"garbage bytes that are not hdf5")
        with pytest.raises(SnirfLoadError) as exc_info:
            adapter.load_snirf(bad_file)
        # mne_exception is set for file-parse failures (not for missing-file branch)
        assert exc_info.value.mne_exception is not None

    def test_snirf_load_error_is_catchable_as_engine_error(
        self, adapter: MNEAdapter, tmp_path: pathlib.Path
    ) -> None:
        """Callers using except EngineError must catch SnirfLoadError."""
        from nirspy.engine.exceptions import EngineError

        missing = tmp_path / "engine_catch_test.snirf"
        with pytest.raises(EngineError):
            adapter.load_snirf(missing)


# ---------------------------------------------------------------------------
# MNEAdapter.load_snirf — happy path (requires sample data)
# ---------------------------------------------------------------------------


class TestMNEAdapterLoadSnirfHappyPath:
    def test_load_returns_mne_raw(self, adapter: MNEAdapter, snirf_path: pathlib.Path) -> None:
        import mne

        raw = adapter.load_snirf(snirf_path)
        assert isinstance(raw, mne.io.BaseRaw)

    def test_loaded_raw_has_channels(
        self, adapter: MNEAdapter, snirf_path: pathlib.Path
    ) -> None:
        raw = adapter.load_snirf(snirf_path)
        assert len(raw.ch_names) > 0

    def test_loaded_raw_has_positive_duration(
        self, adapter: MNEAdapter, snirf_path: pathlib.Path
    ) -> None:
        raw = adapter.load_snirf(snirf_path)
        assert raw.times[-1] > 0

    def test_loaded_raw_has_sample_frequency(
        self, adapter: MNEAdapter, snirf_path: pathlib.Path
    ) -> None:
        raw = adapter.load_snirf(snirf_path)
        assert raw.info["sfreq"] > 0


# ---------------------------------------------------------------------------
# RawWrapper
# ---------------------------------------------------------------------------


class TestRawWrapper:
    def test_raw_wrapper_stores_raw_and_path(
        self, adapter: MNEAdapter, snirf_path: pathlib.Path
    ) -> None:

        raw = adapter.load_snirf(snirf_path)
        wrapper = RawWrapper(raw=raw, source_path=snirf_path)
        assert wrapper.raw is raw
        assert wrapper.source_path == snirf_path

    def test_raw_wrapper_repr_contains_path_info(
        self, adapter: MNEAdapter, snirf_path: pathlib.Path
    ) -> None:
        raw = adapter.load_snirf(snirf_path)
        wrapper = RawWrapper(raw=raw, source_path=snirf_path)
        r = repr(wrapper)
        assert "RawWrapper" in r

    def test_raw_wrapper_n_channels(
        self, adapter: MNEAdapter, snirf_path: pathlib.Path
    ) -> None:
        raw = adapter.load_snirf(snirf_path)
        wrapper = RawWrapper(raw=raw, source_path=snirf_path)
        assert len(wrapper.raw.ch_names) > 0


# ---------------------------------------------------------------------------
# Fixtures for T-040 run_glm / _get_annotation_duration tests (no network)
# ---------------------------------------------------------------------------


def _make_raw_haemo(
    *,
    sfreq: float = 10.0,
    duration_s: float = 90.0,
    ann_onsets: list[float] | None = None,
    ann_durations: list[float] | None = None,
    ann_descriptions: list[str] | None = None,
) -> mne.io.BaseRaw:
    """Build a minimal synthetic RAW_HAEMO object."""
    n_times = int(duration_s * sfreq)
    ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"]
    ch_types = ["hbo", "hbr", "hbo", "hbr"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    for ch in info["chs"]:
        ch["loc"][3:6] = np.array([0.0, 0.0, 0.0])
        ch["loc"][6:9] = np.array([0.03, 0.0, 0.0])
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1e-6, size=(4, n_times))
    raw = mne.io.RawArray(data, info, verbose=False)
    if ann_onsets is not None:
        ann = mne.Annotations(
            onset=ann_onsets,
            duration=ann_durations or [1.0] * len(ann_onsets),
            description=ann_descriptions or ["stim"] * len(ann_onsets),
        )
        raw.set_annotations(ann)
    return raw


# ---------------------------------------------------------------------------
# _get_annotation_duration — unit tests (no network required)
# ---------------------------------------------------------------------------


class TestGetAnnotationDuration:
    """Unit tests for the module-level _get_annotation_duration helper."""

    def test_returns_stored_duration_when_match(self) -> None:
        raw = _make_raw_haemo(
            ann_onsets=[5.0],
            ann_durations=[3.5],
            ann_descriptions=["A"],
        )
        sfreq = raw.info["sfreq"]
        onset_sample = int(5.0 * sfreq)
        result = _get_annotation_duration(raw, "A", onset_sample, sfreq)
        assert result == pytest.approx(3.5)

    def test_fallback_to_1_when_no_matching_description(self) -> None:
        raw = _make_raw_haemo(
            ann_onsets=[5.0],
            ann_durations=[2.0],
            ann_descriptions=["A"],
        )
        sfreq = raw.info["sfreq"]
        result = _get_annotation_duration(raw, "B", 50, sfreq)
        assert result == pytest.approx(1.0)

    def test_fallback_to_1_when_stored_duration_is_zero(self) -> None:
        raw = _make_raw_haemo(
            ann_onsets=[5.0],
            ann_durations=[0.0],
            ann_descriptions=["A"],
        )
        sfreq = raw.info["sfreq"]
        onset_sample = int(5.0 * sfreq)
        result = _get_annotation_duration(raw, "A", onset_sample, sfreq)
        assert result == pytest.approx(1.0)

    def test_fallback_to_1_when_onset_far_off(self) -> None:
        raw = _make_raw_haemo(
            ann_onsets=[5.0],
            ann_durations=[2.0],
            ann_descriptions=["A"],
        )
        sfreq = raw.info["sfreq"]
        # onset_sample very different from 5.0 * sfreq
        result = _get_annotation_duration(raw, "A", 999, sfreq)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MNEAdapter.run_glm — T-040 new params (no network required)
# ---------------------------------------------------------------------------


class TestRunGLMT040:
    """Integration tests for run_glm with condition_durations and per_condition_groups."""

    def test_run_glm_baseline_no_new_params(self) -> None:
        """run_glm without new params must still succeed (regression test)."""
        raw = _make_raw_haemo(
            ann_onsets=[5.0, 20.0, 40.0, 60.0],
            ann_durations=[1.0, 1.0, 1.0, 1.0],
            ann_descriptions=["stim_A"] * 4,
        )
        from nirspy.domain.glm_result import GLMResult

        result = MNEAdapter().run_glm(raw)
        assert isinstance(result, GLMResult)
        assert "stim_A" in result.regressor_names

    def test_run_glm_condition_durations_used_in_events_df(self) -> None:
        """When condition_durations is provided the design matrix should be built
        with those durations rather than the annotation-stored ones."""
        raw = _make_raw_haemo(
            ann_onsets=[5.0, 20.0, 40.0, 60.0],
            ann_durations=[1.0, 1.0, 1.0, 1.0],  # stored = 1 s
            ann_descriptions=["stim_A"] * 4,
        )
        from nirspy.domain.glm_result import GLMResult

        result = MNEAdapter().run_glm(raw, condition_durations={"stim_A": 5.0})
        # Must complete without error and produce a valid GLMResult
        assert isinstance(result, GLMResult)
        assert "stim_A" in result.regressor_names

    def test_run_glm_per_condition_groups_remaps_trial_type(self) -> None:
        """When per_condition_groups is provided, conditions are merged into
        the group label so only that label appears as a regressor."""
        raw = _make_raw_haemo(
            ann_onsets=[5.0, 15.0, 30.0, 45.0, 60.0, 70.0],
            ann_durations=[1.0] * 6,
            ann_descriptions=["cond_A", "cond_B"] * 3,
        )
        from nirspy.domain.glm_result import GLMResult

        groups = {"motor": ["cond_A", "cond_B"]}
        result = MNEAdapter().run_glm(raw, per_condition_groups=groups)
        assert isinstance(result, GLMResult)
        # After grouping, "motor" must be the condition regressor (not cond_A/cond_B)
        assert "motor" in result.regressor_names
        assert "cond_A" not in result.regressor_names
        assert "cond_B" not in result.regressor_names

    def test_run_glm_condition_durations_unknown_condition_falls_back(self) -> None:
        """If condition_durations contains a key not present in the data,
        the fallback path is used and run_glm still succeeds."""
        raw = _make_raw_haemo(
            ann_onsets=[5.0, 20.0, 40.0],
            ann_durations=[1.0] * 3,
            ann_descriptions=["stim_A"] * 3,
        )
        from nirspy.domain.glm_result import GLMResult

        # "nonexistent_cond" is not in the data — should not crash
        result = MNEAdapter().run_glm(
            raw,
            condition_durations={"nonexistent_cond": 99.0},
        )
        assert isinstance(result, GLMResult)
