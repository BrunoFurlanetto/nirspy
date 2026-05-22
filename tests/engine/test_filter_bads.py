"""Tests for MNEAdapter.average_epochs filter_bads flag (ADR-024).

Verifies that:
- filter_bads=True (default) drops bad channels from Evoked
- filter_bads=False preserves all channels including bads
- No-op when no bads exist
- Works with both single Epochs and dict[str, Epochs] paths
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspy.engine.mne_adapter import MNEAdapter


@pytest.fixture()
def adapter() -> MNEAdapter:
    return MNEAdapter()


@pytest.fixture()
def raw_haemo_with_events() -> mne.io.BaseRaw:
    """Synthetic hbo/hbr Raw with annotations for epoching."""
    sfreq = 10.0
    n_times = int(30 * sfreq)  # 30 seconds
    ch_names = [
        "S1_D1 hbo", "S1_D1 hbr",
        "S2_D1 hbo", "S2_D1 hbr",
        "S3_D1 hbo", "S3_D1 hbr",
    ]
    ch_types = ["hbo", "hbr", "hbo", "hbr", "hbo", "hbr"]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(ch_names), n_times)) * 1e-6

    raw = mne.io.RawArray(data, info, verbose=False)

    # Add stimulus annotations
    raw.annotations.append(onset=[2.0, 12.0], duration=[0.0, 0.0],
                           description=["stim", "stim"])
    return raw


class TestFilterBadsDefault:
    """filter_bads=True (default) drops bads from Evoked."""

    def test_bads_excluded_from_evoked(
        self,
        adapter: MNEAdapter,
        raw_haemo_with_events: mne.io.BaseRaw,
    ) -> None:
        raw = raw_haemo_with_events
        # Mark one pair as bad
        raw.info["bads"] = ["S2_D1 hbo", "S2_D1 hbr"]

        epochs = mne.Epochs(
            raw, *mne.events_from_annotations(raw, verbose=False),
            tmin=-1.0, tmax=5.0, baseline=(-1.0, 0.0),
            preload=True, verbose=False,
        )
        result = adapter.average_epochs(epochs)

        for evoked in result.values():
            assert "S2_D1 hbo" not in evoked.ch_names
            assert "S2_D1 hbr" not in evoked.ch_names
            # Good channels remain
            assert "S1_D1 hbo" in evoked.ch_names
            assert len(evoked.info["bads"]) == 0

    def test_channel_count_reduced(
        self,
        adapter: MNEAdapter,
        raw_haemo_with_events: mne.io.BaseRaw,
    ) -> None:
        raw = raw_haemo_with_events
        raw.info["bads"] = ["S2_D1 hbo", "S2_D1 hbr"]

        epochs = mne.Epochs(
            raw, *mne.events_from_annotations(raw, verbose=False),
            tmin=-1.0, tmax=5.0, baseline=(-1.0, 0.0),
            preload=True, verbose=False,
        )
        result = adapter.average_epochs(epochs)

        for evoked in result.values():
            # 6 original - 2 bads = 4
            assert len(evoked.ch_names) == 4


class TestFilterBadsFalse:
    """filter_bads=False preserves all channels."""

    def test_bads_preserved_in_evoked(
        self,
        adapter: MNEAdapter,
        raw_haemo_with_events: mne.io.BaseRaw,
    ) -> None:
        raw = raw_haemo_with_events
        raw.info["bads"] = ["S2_D1 hbo", "S2_D1 hbr"]

        epochs = mne.Epochs(
            raw, *mne.events_from_annotations(raw, verbose=False),
            tmin=-1.0, tmax=5.0, baseline=(-1.0, 0.0),
            preload=True, verbose=False,
        )
        result = adapter.average_epochs(epochs, filter_bads=False)

        for evoked in result.values():
            assert "S2_D1 hbo" in evoked.ch_names
            assert "S2_D1 hbr" in evoked.ch_names
            assert len(evoked.ch_names) == 6


class TestFilterBadsNoBads:
    """No-op when no channels are marked as bad."""

    def test_no_bads_returns_all_channels(
        self,
        adapter: MNEAdapter,
        raw_haemo_with_events: mne.io.BaseRaw,
    ) -> None:
        raw = raw_haemo_with_events
        assert raw.info["bads"] == []

        epochs = mne.Epochs(
            raw, *mne.events_from_annotations(raw, verbose=False),
            tmin=-1.0, tmax=5.0, baseline=(-1.0, 0.0),
            preload=True, verbose=False,
        )
        result = adapter.average_epochs(epochs)

        for evoked in result.values():
            assert len(evoked.ch_names) == 6
            assert len(evoked.info["bads"]) == 0


class TestFilterBadsDictPath:
    """filter_bads works with dict[str, Epochs] (per-condition path)."""

    def test_bads_excluded_dict_path(
        self,
        adapter: MNEAdapter,
        raw_haemo_with_events: mne.io.BaseRaw,
    ) -> None:
        raw = raw_haemo_with_events
        raw.info["bads"] = ["S3_D1 hbo", "S3_D1 hbr"]

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        epochs_dict: dict[str, mne.Epochs] = {}
        for cond, code in event_id.items():
            mask = events[:, 2] == code
            cond_events = events[mask]
            epochs_dict[cond] = mne.Epochs(
                raw, cond_events, event_id={cond: code},
                tmin=-1.0, tmax=5.0, baseline=(-1.0, 0.0),
                preload=True, verbose=False,
            )

        result = adapter.average_epochs(epochs_dict, filter_bads=True)

        for evoked in result.values():
            assert "S3_D1 hbo" not in evoked.ch_names
            assert len(evoked.ch_names) == 4

    def test_bads_preserved_dict_path(
        self,
        adapter: MNEAdapter,
        raw_haemo_with_events: mne.io.BaseRaw,
    ) -> None:
        raw = raw_haemo_with_events
        raw.info["bads"] = ["S3_D1 hbo", "S3_D1 hbr"]

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        epochs_dict: dict[str, mne.Epochs] = {}
        for cond, code in event_id.items():
            mask = events[:, 2] == code
            cond_events = events[mask]
            epochs_dict[cond] = mne.Epochs(
                raw, cond_events, event_id={cond: code},
                tmin=-1.0, tmax=5.0, baseline=(-1.0, 0.0),
                preload=True, verbose=False,
            )

        result = adapter.average_epochs(epochs_dict, filter_bads=False)

        for evoked in result.values():
            assert "S3_D1 hbo" in evoked.ch_names
            assert len(evoked.ch_names) == 6
