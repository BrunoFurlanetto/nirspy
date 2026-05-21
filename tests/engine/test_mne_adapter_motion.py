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
