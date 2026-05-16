"""Shared fixtures for blocks tests -- synthetic fNIRS data."""

from __future__ import annotations

import numpy as np
import pytest

import mne


@pytest.fixture()
def raw_cw_amplitude() -> mne.io.BaseRaw:
    """Create a synthetic Raw with fnirs_cw_amplitude channels.

    Two source-detector pairs at 760nm and 850nm (4 channels total),
    10 seconds at 10 Hz sampling rate.
    """
    sfreq = 10.0
    n_times = int(10 * sfreq)
    n_channels = 4

    # Create channel info for fNIRS CW amplitude
    ch_names = ["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"]
    ch_types = ["fnirs_cw_amplitude"] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Set source/detector positions (required for OD conversion)
    # Simple 2D montage on the scalp
    import numpy as np
    sources = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
    detectors = np.array([[0.015, 0.0, 0.0]])
    # Assign source/detector/wavelength info
    for i, ch in enumerate(info["chs"]):
        src_idx = i // 2
        ch["loc"][3:6] = sources[src_idx]
        ch["loc"][6:9] = detectors[0]
        # Set wavelength in loc[9]
        wavelength = 760.0 if i % 2 == 0 else 850.0
        ch["loc"][9] = wavelength

    # Generate positive intensity data (required for log-ratio in OD)
    rng = np.random.default_rng(42)
    data = rng.uniform(low=0.5, high=2.0, size=(n_channels, n_times))

    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture()
def raw_od(raw_cw_amplitude: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Raw with fnirs_od channels (post optical_density conversion)."""
    from nirspy.engine.mne_adapter import MNEAdapter
    adapter = MNEAdapter()
    return adapter.raw_to_od(raw_cw_amplitude)


@pytest.fixture()
def raw_haemo(raw_od: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Raw with hbo/hbr channels (post Beer-Lambert conversion)."""
    from nirspy.engine.mne_adapter import MNEAdapter
    adapter = MNEAdapter()
    return adapter.beer_lambert(raw_od)
