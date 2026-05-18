"""Regression tests for average_epochs handling of empty conditions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nirspy.engine.exceptions import MNEOperationError
from nirspy.engine.mne_adapter import MNEAdapter


def _make_epochs(condition_lengths: dict[str, int]) -> MagicMock:
    """Build a mock epochs object indexable by condition with given length."""
    epochs = MagicMock()
    epochs.event_id = dict.fromkeys(condition_lengths, 1)

    def _getitem(condition: str) -> MagicMock:
        cond = MagicMock()
        cond.__len__.return_value = condition_lengths[condition]
        evoked = MagicMock(name=f"evoked_{condition}")
        cond.average.return_value = evoked
        return cond

    epochs.__getitem__.side_effect = _getitem
    return epochs


def test_skips_empty_conditions_and_keeps_non_empty() -> None:
    """Conditions whose epochs were all rejected are skipped, not errors."""
    adapter = MNEAdapter()
    epochs = _make_epochs({"A": 3, "B": 0, "C": 1, "D": 0})

    result = adapter.average_epochs(epochs)

    assert set(result.keys()) == {"A", "C"}


def test_all_empty_raises_mne_operation_error() -> None:
    """When every condition is empty, a clear error is raised."""
    adapter = MNEAdapter()
    epochs = _make_epochs({"A": 0, "B": 0, "C": 0})

    with pytest.raises(MNEOperationError, match="all conditions"):
        adapter.average_epochs(epochs)


def test_no_empty_conditions_returns_all() -> None:
    """Happy path: every condition has epochs, all are averaged."""
    adapter = MNEAdapter()
    epochs = _make_epochs({"A": 2, "B": 5})

    result = adapter.average_epochs(epochs)

    assert set(result.keys()) == {"A", "B"}
