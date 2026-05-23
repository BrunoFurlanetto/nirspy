"""Tests for MNEAdapter.create_epochs_per_group with event_indices (T-030).

Uses a synthetic mne.io.RawArray with injected annotations so the tests run
without network access and finish in <100 ms each.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nirspy.blocks.analysis import ConditionGroup
from nirspy.engine.mne_adapter import MNEAdapter

# ---------------------------------------------------------------------------
# Synthetic Raw fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> MNEAdapter:
    return MNEAdapter()


@pytest.fixture
def synthetic_raw_haemo() -> Any:
    """Create a synthetic mne.io.RawArray with hbo channels + annotations.

    Annotations (sorted by onset, i.e. chronological event_index order):
        idx 0 : "S1" @ 5s
        idx 1 : "S2" @ 15s
        idx 2 : "S1" @ 25s
        idx 3 : "S2" @ 35s
        idx 4 : "S1" @ 45s

    The fixture creates a 60-second signal at 10 Hz with 4 hbo channels.
    """
    import mne
    from mne import Annotations

    sfreq = 10.0
    duration = 60.0
    n_times = int(sfreq * duration)
    n_ch = 4
    data = np.random.default_rng(42).normal(0, 1e-6, (n_ch, n_times))

    ch_names = ["S1_D1 hbo", "S1_D2 hbo", "S2_D1 hbo", "S2_D2 hbo"]
    ch_types = ["hbo"] * n_ch
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)

    # Inject annotations: onsets, durations, descriptions
    onsets = [5.0, 15.0, 25.0, 35.0, 45.0]
    durations = [1.0] * 5
    descriptions = ["S1", "S2", "S1", "S2", "S1"]
    annotations = Annotations(onsets, durations, descriptions)
    raw.set_annotations(annotations)
    return raw


# ---------------------------------------------------------------------------
# _events_by_indices unit tests
# ---------------------------------------------------------------------------


class TestEventsByIndices:
    def test_single_index_returns_one_event(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        events, event_id = adapter._events_by_indices(
            synthetic_raw_haemo, [0]
        )
        assert events.shape == (1, 3)
        assert "S1" in event_id

    def test_multiple_indices_return_correct_count(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        # indices 0, 2, 4 -> all "S1" occurrences
        events, event_id = adapter._events_by_indices(
            synthetic_raw_haemo, [0, 2, 4]
        )
        assert events.shape[0] == 3
        assert "S1" in event_id

    def test_mixed_conditions(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        # indices 0 (S1@5s), 1 (S2@15s), 3 (S2@35s)
        events, event_id = adapter._events_by_indices(
            synthetic_raw_haemo, [0, 1, 3]
        )
        assert events.shape[0] == 3
        assert "S1" in event_id
        assert "S2" in event_id

    def test_out_of_range_index_skipped(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        events, event_id = adapter._events_by_indices(
            synthetic_raw_haemo, [0, 999]  # 999 is out of range
        )
        assert events.shape[0] == 1  # only index 0 valid

    def test_empty_indices_returns_empty(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        events, event_id = adapter._events_by_indices(
            synthetic_raw_haemo, []
        )
        assert events.shape == (0, 3)
        assert event_id == {}

    def test_events_sorted_by_sample(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        # Pass indices out of natural order -> output must still be sorted
        events, _ = adapter._events_by_indices(
            synthetic_raw_haemo, [4, 0, 2]  # reversed
        )
        assert events.shape[0] == 3
        assert list(events[:, 0]) == sorted(events[:, 0].tolist())


# ---------------------------------------------------------------------------
# create_epochs_per_group — event_indices mode
# ---------------------------------------------------------------------------


class TestCreateEpochsPerGroupEventIndices:
    def test_creates_epochs_for_event_indices_group(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        """Group using event_indices produces Epochs with correct count."""
        import mne

        group_a = ConditionGroup(
            label="GroupA",
            event_indices=[0, 2],  # two S1 occurrences
            tmin=-1.0, tmax=5.0,
            baseline_tmin=-1.0, baseline_tmax=0.0,
        )
        result = adapter.create_epochs_per_group(
            synthetic_raw_haemo,
            groups={"GroupA": group_a},
            reject=None,
        )
        assert "GroupA" in result
        assert isinstance(result["GroupA"], mne.Epochs)
        assert len(result["GroupA"]) == 2

    def test_event_indices_only_selected_occurrences(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        """Selecting 1 out of 3 S1 occurrences gives exactly 1 epoch."""

        group_a = ConditionGroup(
            label="G",
            event_indices=[2],  # only the second S1 (idx 2 = onset 25s)
            tmin=-1.0, tmax=5.0,
            baseline_tmin=-1.0, baseline_tmax=0.0,
        )
        result = adapter.create_epochs_per_group(
            synthetic_raw_haemo,
            groups={"G": group_a},
            reject=None,
        )
        assert "G" in result
        assert len(result["G"]) == 1

    def test_condition_names_mode_unchanged(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        """condition_names mode still works exactly as before (T-024)."""

        group_s1 = ConditionGroup(
            label="all_S1",
            condition_names=["S1"],
            tmin=-1.0, tmax=5.0,
            baseline_tmin=-1.0, baseline_tmax=0.0,
        )
        result = adapter.create_epochs_per_group(
            synthetic_raw_haemo,
            groups={"all_S1": group_s1},
            reject=None,
        )
        assert "all_S1" in result
        # S1 appears 3 times -> 3 epochs
        assert len(result["all_S1"]) == 3

    def test_multiple_groups_mixed_modes(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        """Two groups: one condition_names, one event_indices -> both work."""

        group_names = ConditionGroup(
            label="by_name",
            condition_names=["S2"],
            tmin=-1.0, tmax=5.0,
            baseline_tmin=-1.0, baseline_tmax=0.0,
        )
        group_indices = ConditionGroup(
            label="by_idx",
            event_indices=[0, 4],  # S1 idx 0 and S1 idx 4
            tmin=-1.0, tmax=5.0,
            baseline_tmin=-1.0, baseline_tmax=0.0,
        )
        result = adapter.create_epochs_per_group(
            synthetic_raw_haemo,
            groups={"by_name": group_names, "by_idx": group_indices},
            reject=None,
        )
        assert "by_name" in result
        assert len(result["by_name"]) == 2  # S2 appears twice

        assert "by_idx" in result
        assert len(result["by_idx"]) == 2  # two S1 occurrences

    def test_all_invalid_indices_returns_empty_group(
        self, adapter: MNEAdapter, synthetic_raw_haemo: Any
    ) -> None:
        """Group with only out-of-range indices is skipped (no key in result)."""
        group = ConditionGroup(
            label="bad",
            event_indices=[100, 200],
            tmin=-1.0, tmax=5.0,
            baseline_tmin=-1.0, baseline_tmax=0.0,
        )
        result = adapter.create_epochs_per_group(
            synthetic_raw_haemo,
            groups={"bad": group},
            reject=None,
        )
        assert "bad" not in result
