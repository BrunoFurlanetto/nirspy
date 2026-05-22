"""Tests for MNEAdapter motion correction methods (T-015)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from nirspy.engine.exceptions import MNEOperationError
from nirspy.engine.mne_adapter import MNEAdapter


class TestMNEAdapterTDDR:
    """Tests for MNEAdapter.tddr()."""

    @pytest.fixture()
    def adapter(self) -> MNEAdapter:
        return MNEAdapter()

    @pytest.fixture()
    def raw_od(self) -> mne.io.BaseRaw:
        """Minimal OD raw for adapter tests."""
        sfreq = 10.0
        n_times = int(10 * sfreq)
        ch_names = ["S1_D1 760", "S1_D1 850"]
        ch_types = ["fnirs_od"] * 2
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        sources = [[0.0, 0.0, 0.0]]
        detectors = [[0.03, 0.0, 0.0]]
        for i, ch in enumerate(info["chs"]):
            ch["loc"][3:6] = sources[0]
            ch["loc"][6:9] = detectors[0]
            ch["loc"][9] = 760.0 if i % 2 == 0 else 850.0

        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, n_times))
        return mne.io.RawArray(data, info, verbose=False)

    def test_adapter_tddr_returns_raw(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """tddr() returns a BaseRaw object."""
        result = adapter.tddr(raw_od)
        assert isinstance(result, mne.io.BaseRaw)

    def test_adapter_tddr_preserves_channels(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """tddr() preserves channel names and count."""
        result = adapter.tddr(raw_od)
        assert result.ch_names == raw_od.ch_names

    def test_adapter_tddr_wraps_error(self, adapter: MNEAdapter) -> None:
        """tddr() wraps exceptions in MNEOperationError."""
        bad_raw = MagicMock(spec=mne.io.BaseRaw)
        with pytest.raises(MNEOperationError, match="tddr"):
            adapter.tddr(bad_raw)

    def test_adapter_tddr_delegates_to_mne(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """tddr() delegates to mne.preprocessing.nirs.temporal_derivative_distribution_repair."""
        mock_tddr = MagicMock(return_value=raw_od)
        with patch(
            "mne.preprocessing.nirs.temporal_derivative_distribution_repair",
            mock_tddr,
        ):
            result = adapter.tddr(raw_od)
        mock_tddr.assert_called_once_with(raw_od)
        assert result is raw_od


class TestMNEAdapterSpline:
    """Tests for MNEAdapter.spline_motion_correction() (T-016)."""

    @pytest.fixture()
    def adapter(self) -> MNEAdapter:
        return MNEAdapter()

    @pytest.fixture()
    def raw_od(self) -> mne.io.BaseRaw:
        """Minimal OD raw for adapter tests."""
        sfreq = 10.0
        n_times = int(10 * sfreq)
        ch_names = ["S1_D1 760", "S1_D1 850"]
        ch_types = ["fnirs_od"] * 2
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        sources = [[0.0, 0.0, 0.0]]
        detectors = [[0.03, 0.0, 0.0]]
        for i, ch in enumerate(info["chs"]):
            ch["loc"][3:6] = sources[0]
            ch["loc"][6:9] = detectors[0]
            ch["loc"][9] = 760.0 if i % 2 == 0 else 850.0

        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, n_times))
        return mne.io.RawArray(data, info, verbose=False)

    def test_adapter_spline_returns_raw(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """spline_motion_correction() returns a BaseRaw object."""
        result = adapter.spline_motion_correction(raw_od)
        assert isinstance(result, mne.io.BaseRaw)

    def test_adapter_spline_preserves_channels(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """spline_motion_correction() preserves channel names and count."""
        result = adapter.spline_motion_correction(raw_od)
        assert result.ch_names == raw_od.ch_names

    def test_adapter_spline_detects_artifact(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """Spline detects and corrects a synthetic spike artifact."""
        raw_copy = raw_od.copy()
        data = raw_copy.get_data()
        # Inject large spike (will have high z-score in temporal derivative)
        data[0, 50] += 10.0
        raw_copy._data = data

        result = adapter.spline_motion_correction(raw_copy, threshold=2.0)
        out_data = result.get_data()
        # The spike should be attenuated (not necessarily identical to original)
        assert not np.array_equal(data, out_data)
        # The corrected value at the spike should be closer to neighbours
        original_spike = data[0, 50]
        corrected_spike = out_data[0, 50]
        neighbour_mean = (data[0, 49] + data[0, 51]) / 2
        assert abs(corrected_spike - neighbour_mean) < abs(
            original_spike - neighbour_mean
        )

    def test_adapter_spline_threshold_sensitivity(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """Higher threshold = fewer corrections (more permissive)."""
        raw_copy = raw_od.copy()
        data = raw_copy.get_data()
        data[0, 50] += 5.0  # moderate spike
        raw_copy._data = data

        result_low = adapter.spline_motion_correction(
            raw_copy.copy(), threshold=1.0
        )
        result_high = adapter.spline_motion_correction(
            raw_copy.copy(), threshold=20.0
        )
        out_low = result_low.get_data()
        out_high = result_high.get_data()

        # With very high threshold, output should be closer to input
        diff_low = np.sum(np.abs(data - out_low))
        diff_high = np.sum(np.abs(data - out_high))
        assert diff_high <= diff_low

    def test_adapter_spline_wraps_error(self, adapter: MNEAdapter) -> None:
        """spline_motion_correction() wraps exceptions in MNEOperationError."""
        bad_raw = MagicMock(spec=mne.io.BaseRaw)
        bad_raw.copy.side_effect = RuntimeError("copy failed")
        with pytest.raises(MNEOperationError, match="spline_motion_correction"):
            adapter.spline_motion_correction(bad_raw)

    def test_adapter_spline_short_signal(self, adapter: MNEAdapter) -> None:
        """spline_motion_correction() handles very short signals gracefully."""
        ch_names = ["S1_D1 760", "S1_D1 850"]
        ch_types = ["fnirs_od"] * 2
        info = mne.create_info(ch_names=ch_names, sfreq=10.0, ch_types=ch_types)
        for i, ch in enumerate(info["chs"]):
            ch["loc"][3:6] = [0.0, 0.0, 0.0]
            ch["loc"][6:9] = [0.03, 0.0, 0.0]
            ch["loc"][9] = 760.0 if i % 2 == 0 else 850.0
        data = np.zeros((2, 3))  # only 3 samples
        raw = mne.io.RawArray(data, info, verbose=False)
        result = adapter.spline_motion_correction(raw)
        assert isinstance(result, mne.io.BaseRaw)


class TestMNEAdapterWavelet:
    """Tests for MNEAdapter.wavelet_motion_correction() (T-017)."""

    @pytest.fixture()
    def adapter(self) -> MNEAdapter:
        return MNEAdapter()

    @pytest.fixture()
    def raw_od(self) -> mne.io.BaseRaw:
        """Minimal OD raw for adapter tests."""
        sfreq = 10.0
        n_times = int(10 * sfreq)
        ch_names = ["S1_D1 760", "S1_D1 850"]
        ch_types = ["fnirs_od"] * 2
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        sources = [[0.0, 0.0, 0.0]]
        detectors = [[0.03, 0.0, 0.0]]
        for i, ch in enumerate(info["chs"]):
            ch["loc"][3:6] = sources[0]
            ch["loc"][6:9] = detectors[0]
            ch["loc"][9] = 760.0 if i % 2 == 0 else 850.0

        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, n_times))
        return mne.io.RawArray(data, info, verbose=False)

    def test_adapter_wavelet_returns_raw(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """wavelet_motion_correction() returns a BaseRaw object."""
        result = adapter.wavelet_motion_correction(raw_od)
        assert isinstance(result, mne.io.BaseRaw)

    def test_adapter_wavelet_preserves_channels(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """wavelet_motion_correction() preserves channel names and count."""
        result = adapter.wavelet_motion_correction(raw_od)
        assert result.ch_names == raw_od.ch_names

    def test_adapter_wavelet_denoises_signal(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """Wavelet denoises a synthetic signal with injected spike."""
        raw_copy = raw_od.copy()
        data = raw_copy.get_data()
        # Inject large spike artifact
        data[0, 50] += 10.0
        raw_copy._data = data

        result = adapter.wavelet_motion_correction(raw_copy, iqr_multiplier=0.5)
        out_data = result.get_data()
        # The spike should be attenuated
        assert not np.array_equal(data, out_data)
        original_spike = data[0, 50]
        corrected_spike = out_data[0, 50]
        neighbour_mean = (data[0, 49] + data[0, 51]) / 2
        assert abs(corrected_spike - neighbour_mean) < abs(
            original_spike - neighbour_mean
        )

    def test_adapter_wavelet_iqr_sensitivity(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """Higher iqr_multiplier = more smoothing (stronger correction).

        Soft-threshold zeros coefficients below threshold.  Higher
        multiplier -> higher threshold -> more coefficients zeroed ->
        larger deviation from original.
        """
        raw_copy = raw_od.copy()
        data = raw_copy.get_data()

        result_low = adapter.wavelet_motion_correction(
            raw_copy.copy(), iqr_multiplier=0.1
        )
        result_high = adapter.wavelet_motion_correction(
            raw_copy.copy(), iqr_multiplier=10.0
        )
        out_low = result_low.get_data()
        out_high = result_high.get_data()

        # Higher multiplier should modify the signal more
        diff_low = np.sum(np.abs(data - out_low))
        diff_high = np.sum(np.abs(data - out_high))
        assert diff_high >= diff_low

    def test_adapter_wavelet_invalid_wavelet_raises(
        self, adapter: MNEAdapter, raw_od: mne.io.BaseRaw
    ) -> None:
        """wavelet_motion_correction() raises MNEOperationError for invalid wavelet."""
        with pytest.raises(MNEOperationError, match="wavelet_motion_correction"):
            adapter.wavelet_motion_correction(raw_od, wavelet="not_a_wavelet_xyz")

    def test_adapter_wavelet_wraps_error(self, adapter: MNEAdapter) -> None:
        """wavelet_motion_correction() wraps exceptions in MNEOperationError."""
        bad_raw = MagicMock(spec=mne.io.BaseRaw)
        bad_raw.copy.side_effect = RuntimeError("copy failed")
        with pytest.raises(MNEOperationError, match="wavelet_motion_correction"):
            adapter.wavelet_motion_correction(bad_raw)

    def test_adapter_wavelet_short_signal(self, adapter: MNEAdapter) -> None:
        """wavelet_motion_correction() handles very short signals gracefully."""
        ch_names = ["S1_D1 760", "S1_D1 850"]
        ch_types = ["fnirs_od"] * 2
        info = mne.create_info(ch_names=ch_names, sfreq=10.0, ch_types=ch_types)
        for i, ch in enumerate(info["chs"]):
            ch["loc"][3:6] = [0.0, 0.0, 0.0]
            ch["loc"][6:9] = [0.03, 0.0, 0.0]
            ch["loc"][9] = 760.0 if i % 2 == 0 else 850.0
        data = np.zeros((2, 3))  # only 3 samples
        raw = mne.io.RawArray(data, info, verbose=False)
        result = adapter.wavelet_motion_correction(raw)
        assert isinstance(result, mne.io.BaseRaw)
