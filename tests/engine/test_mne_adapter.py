"""Tests for MNEAdapter and RawWrapper.

Integration tests that require MNE-NIRS sample data.
Skipped automatically when the dataset cannot be downloaded.
"""

from __future__ import annotations

import pathlib

import pytest

from nirspy.engine.exceptions import SnirfLoadError
from nirspy.engine.mne_adapter import MNEAdapter, RawWrapper


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
        import mne

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
