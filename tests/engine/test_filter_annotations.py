"""Tests for MNEAdapter.filter_annotations_by_conditions (T-042)."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspy.domain.conditions import ConditionConfig, GlobalConditions
from nirspy.engine.mne_adapter import MNEAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_with_two_conditions() -> mne.io.BaseRaw:
    """Synthetic Raw with two conditions, multiple occurrences each.

    Annotations:
        "1.0"  at t=10, 45, 80  (3 occurrences)
        "2.0"  at t=30, 70      (2 occurrences)
    """
    sfreq = 10.0
    data = np.zeros((1, int(sfreq * 200)))
    info = mne.create_info(["Fz"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(data, info, verbose=False)
    onsets = [10.0, 45.0, 80.0, 30.0, 70.0]
    durations = [1.0] * 5
    descs = ["1.0", "1.0", "1.0", "2.0", "2.0"]
    raw.set_annotations(mne.Annotations(onsets, durations, descs))
    return raw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFilterAnnotationsByConditions:
    """Tests for MNEAdapter.filter_annotations_by_conditions."""

    def test_filter_annotations_renames_condition(
        self, raw_with_two_conditions: mne.io.BaseRaw
    ) -> None:
        """Annotation '1.0' is renamed to 'Cognitive' per ConditionConfig."""
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    duration=1.0,
                ),
            )
        )
        raw_out = MNEAdapter.filter_annotations_by_conditions(
            raw_with_two_conditions, gc
        )
        descriptions = list(raw_out.annotations.description)
        assert "Cognitive" in descriptions
        assert "1.0" not in descriptions

    def test_filter_annotations_excludes_occurrence(
        self, raw_with_two_conditions: mne.io.BaseRaw
    ) -> None:
        """When included_occurrences=(0, 2), occurrence index 1 is excluded."""
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    included_occurrences=(0, 2),
                    duration=1.0,
                ),
            )
        )
        raw_out = MNEAdapter.filter_annotations_by_conditions(
            raw_with_two_conditions, gc
        )
        cognitive_onsets = [
            float(o)
            for o, d in zip(
                raw_out.annotations.onset,
                raw_out.annotations.description,
                strict=False,
            )
            if d == "Cognitive"
        ]
        # Original onsets for "1.0": 10, 45, 80 → idx 0=10, idx 1=45 (excluded), idx 2=80
        assert 10.0 in cognitive_onsets
        assert 80.0 in cognitive_onsets
        assert 45.0 not in cognitive_onsets

    def test_filter_annotations_all_occurrences_when_none(
        self, raw_with_two_conditions: mne.io.BaseRaw
    ) -> None:
        """included_occurrences=None → all occurrences are kept."""
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    included_occurrences=None,
                    duration=1.0,
                ),
            )
        )
        raw_out = MNEAdapter.filter_annotations_by_conditions(
            raw_with_two_conditions, gc
        )
        cognitive_count = sum(
            1 for d in raw_out.annotations.description if d == "Cognitive"
        )
        assert cognitive_count == 3

    def test_filter_annotations_updates_duration(
        self, raw_with_two_conditions: mne.io.BaseRaw
    ) -> None:
        """Annotation duration is overridden by ConditionConfig.duration."""
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    duration=5.0,  # override original 1.0
                ),
            )
        )
        raw_out = MNEAdapter.filter_annotations_by_conditions(
            raw_with_two_conditions, gc
        )
        for onset, dur, desc in zip(
            raw_out.annotations.onset,
            raw_out.annotations.duration,
            raw_out.annotations.description,
            strict=False,
        ):
            if desc == "Cognitive":
                assert dur == pytest.approx(5.0), (
                    f"Expected duration 5.0 for Cognitive at onset {onset}, got {dur}"
                )

    def test_filter_annotations_oob_index_ignored(
        self, raw_with_two_conditions: mne.io.BaseRaw
    ) -> None:
        """Out-of-range occurrence indices are silently ignored (no crash)."""
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    # "1.0" has only 3 occurrences (idx 0,1,2); idx 99 is OOB
                    included_occurrences=(0, 99),
                    duration=1.0,
                ),
            )
        )
        # Must not raise; just keep occurrence idx=0 and silently skip idx=99
        raw_out = MNEAdapter.filter_annotations_by_conditions(
            raw_with_two_conditions, gc
        )
        cognitive_count = sum(
            1 for d in raw_out.annotations.description if d == "Cognitive"
        )
        # Only idx=0 is valid; result must have exactly 1 occurrence
        assert cognitive_count == 1

    def test_filter_annotations_unmatched_kept_unchanged(
        self, raw_with_two_conditions: mne.io.BaseRaw
    ) -> None:
        """Annotations not matched by any ConditionConfig are kept unchanged."""
        # Only map "1.0" → "Cognitive"; "2.0" has no mapping
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    duration=1.0,
                ),
            )
        )
        raw_out = MNEAdapter.filter_annotations_by_conditions(
            raw_with_two_conditions, gc
        )
        descriptions = list(raw_out.annotations.description)
        # "2.0" annotations must still be present
        assert "2.0" in descriptions
        # "1.0" must have been renamed
        assert "1.0" not in descriptions

    def test_filter_annotations_does_not_mutate_original(
        self, raw_with_two_conditions: mne.io.BaseRaw
    ) -> None:
        """filter_annotations_by_conditions returns a copy and does not mutate input."""
        original_descs = list(raw_with_two_conditions.annotations.description)
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    duration=1.0,
                ),
            )
        )
        MNEAdapter.filter_annotations_by_conditions(raw_with_two_conditions, gc)

        assert list(raw_with_two_conditions.annotations.description) == original_descs
