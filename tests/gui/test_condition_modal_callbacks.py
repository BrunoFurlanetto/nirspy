"""Regression tests for condition-config modal callbacks (T-042).

Covers:
- apply_condition_config: duration persistence from DOM inputs, fallback chain
  (DOM → prev_gc_store → state), validation errors, multi-condition, groups.
- open_condition_modal_from_button: restore last-applied values from
  global-conditions-store, occurrences preservation, groups restoration.
- cancel_condition_config: closes modal, no_update on n_clicks=None.

These tests call the callback functions directly (no Dash server required).
The `no_update` sentinel from Dash is compared by identity.
"""

from __future__ import annotations

from typing import Any

import pytest
from dash import no_update

from nirspy.gui.callbacks.pipeline_callbacks import (
    apply_condition_config,
    cancel_condition_config,
    open_condition_modal_from_button,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cond(
    name: str = "HbO",
    original_name: str | None = None,
    duration: float = 1.0,
    tmin: float = -2.0,
    tmax: float = 18.0,
    baseline_tmin: float = -2.0,
    baseline_tmax: float = 0.0,
    occurrences: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "original_name": original_name if original_name is not None else name,
        "duration": duration,
        "tmin": tmin,
        "tmax": tmax,
        "baseline_tmin": baseline_tmin,
        "baseline_tmax": baseline_tmax,
        "occurrences": occurrences or [],
    }


def _state(
    conditions: list[dict[str, Any]] | None = None,
    groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    s: dict[str, Any] = {"conditions": conditions or []}
    if groups is not None:
        s["groups"] = groups
    return s


def _gc_store(
    conditions: list[dict[str, Any]] | None = None,
    groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Minimal serialised GlobalConditions dict (as produced by global_conditions_to_dict)."""
    result: dict[str, Any] = {"conditions": conditions or []}
    if groups is not None:
        result["groups"] = groups
    return result


def _gc_cond(
    name: str = "HbO",
    original_name: str | None = None,
    duration: float = 1.0,
    tmin: float = -2.0,
    tmax: float = 18.0,
    baseline_tmin: float = -2.0,
    baseline_tmax: float = 0.0,
    included_occurrences: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "original_name": original_name if original_name is not None else name,
        "duration": duration,
        "tmin": tmin,
        "tmax": tmax,
        "baseline_tmin": baseline_tmin,
        "baseline_tmax": baseline_tmax,
        "included_occurrences": included_occurrences,
    }


# ---------------------------------------------------------------------------
# TestApplyConditionConfig
# ---------------------------------------------------------------------------


class TestApplyConditionConfig:
    """Tests for apply_condition_config callback."""

    # A-01 --------------------------------------------------------------------
    def test_dom_duration_persisted_to_store(self) -> None:
        """DOM value [7.5] → store condition duration == 7.5."""
        state = _state(conditions=[_cond("HbO", duration=1.0)])
        result = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[7.5],
            prev_gc=None,
        )
        store_data, is_open, warning, warning_style, synced_state = result
        assert store_data is not no_update
        assert store_data["conditions"][0]["duration"] == 7.5
        assert is_open is False

    # A-02 --------------------------------------------------------------------
    def test_second_apply_overwrites_duration(self) -> None:
        """Second Apply with dom=[12.0] overwrites previous duration."""
        state = _state(conditions=[_cond("HbO", duration=7.5)])
        result = apply_condition_config(
            n_clicks=2,
            state=state,
            dom_durations=[12.0],
            prev_gc=None,
        )
        store_data, is_open, *_ = result
        assert store_data["conditions"][0]["duration"] == 12.0

    # A-03 --------------------------------------------------------------------
    def test_fallback_to_prev_gc_when_dom_none(self) -> None:
        """dom=None → falls back to prev_gc duration (15.0)."""
        prev_gc = _gc_store(conditions=[_gc_cond("HbO", duration=15.0)])
        state = _state(conditions=[_cond("HbO", duration=1.0)])
        result = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[None],
            prev_gc=prev_gc,
        )
        store_data, is_open, *_ = result
        assert store_data["conditions"][0]["duration"] == 15.0

    # A-04 --------------------------------------------------------------------
    def test_fallback_to_state_when_dom_none_and_prev_gc_none(self) -> None:
        """dom=None, prev_gc=None → falls back to state duration (10.0)."""
        state = _state(conditions=[_cond("HbO", duration=10.0)])
        result = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[None],
            prev_gc=None,
        )
        store_data, is_open, *_ = result
        assert store_data["conditions"][0]["duration"] == 10.0

    # A-05 --------------------------------------------------------------------
    def test_modal_closes_on_success(self) -> None:
        """is_open == False on successful Apply."""
        state = _state(conditions=[_cond("HbO", duration=5.0)])
        _, is_open, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[5.0],
            prev_gc=None,
        )
        assert is_open is False

    # A-06 --------------------------------------------------------------------
    def test_tmin_tmax_preserved_in_store(self) -> None:
        """tmin/tmax from state are preserved in the store."""
        state = _state(conditions=[_cond("HbO", duration=5.0, tmin=-3.0, tmax=20.0)])
        store_data, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[5.0],
            prev_gc=None,
        )
        assert store_data["conditions"][0]["tmin"] == -3.0
        assert store_data["conditions"][0]["tmax"] == 20.0

    # A-07 --------------------------------------------------------------------
    def test_baseline_tmin_tmax_preserved(self) -> None:
        """baseline_tmin/baseline_tmax from state are preserved in the store."""
        state = _state(
            conditions=[_cond("HbO", duration=5.0, baseline_tmin=-1.5, baseline_tmax=0.5)]
        )
        store_data, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[5.0],
            prev_gc=None,
        )
        assert store_data["conditions"][0]["baseline_tmin"] == -1.5
        assert store_data["conditions"][0]["baseline_tmax"] == 0.5

    # A-08 --------------------------------------------------------------------
    def test_modal_stays_open_when_tmin_equals_tmax(self) -> None:
        """tmin == tmax is invalid → modal stays open."""
        state = _state(conditions=[_cond("HbO", duration=5.0, tmin=0.0, tmax=0.0)])
        _, is_open, warning, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[5.0],
            prev_gc=None,
        )
        assert is_open is True
        assert warning  # non-empty error message

    # A-09 --------------------------------------------------------------------
    def test_modal_stays_open_when_duration_zero(self) -> None:
        """duration == 0.0 is invalid → modal stays open."""
        state = _state(conditions=[_cond("HbO", duration=5.0)])
        _, is_open, warning, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[0.0],
            prev_gc=None,
        )
        assert is_open is True
        assert warning

    # A-10 --------------------------------------------------------------------
    def test_modal_stays_open_when_duration_negative(self) -> None:
        """duration < 0 is invalid → modal stays open."""
        state = _state(conditions=[_cond("HbO", duration=5.0)])
        _, is_open, warning, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[-1.0],
            prev_gc=None,
        )
        assert is_open is True
        assert warning

    # A-11 --------------------------------------------------------------------
    def test_no_update_when_n_clicks_none(self) -> None:
        """n_clicks=None → all 5 outputs return no_update."""
        state = _state(conditions=[_cond("HbO")])
        result = apply_condition_config(
            n_clicks=None,
            state=state,
            dom_durations=[5.0],
            prev_gc=None,
        )
        assert all(r is no_update for r in result)

    # A-12 --------------------------------------------------------------------
    def test_no_update_when_state_none(self) -> None:
        """state=None → all 5 outputs return no_update."""
        result = apply_condition_config(
            n_clicks=1,
            state=None,
            dom_durations=[5.0],
            prev_gc=None,
        )
        assert all(r is no_update for r in result)

    # A-13 --------------------------------------------------------------------
    def test_warning_when_conditions_empty(self) -> None:
        """conditions=[] → warning 'at least one condition' and modal stays open."""
        state = _state(conditions=[])
        _, is_open, warning, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[],
            prev_gc=None,
        )
        assert is_open is True
        assert "least one condition" in warning.lower()

    # A-14 --------------------------------------------------------------------
    def test_multiple_conditions_each_gets_own_dom_duration(self) -> None:
        """Two conditions [7.0, 9.0] → each stored with its own duration."""
        state = _state(
            conditions=[
                _cond("HbO", duration=1.0),
                _cond("HbR", duration=1.0),
            ]
        )
        store_data, is_open, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[7.0, 9.0],
            prev_gc=None,
        )
        assert is_open is False
        durations = {c["original_name"]: c["duration"] for c in store_data["conditions"]}
        assert durations["HbO"] == 7.0
        assert durations["HbR"] == 9.0

    # A-15 --------------------------------------------------------------------
    def test_dom_duration_index_aligned_with_raw_conditions(self) -> None:
        """dom_durations index must match raw_conditions order, not condition_configs order.

        Regression for the bug where _cond_idx = len(condition_configs) was used
        instead of the raw index — causing wrong DOM value to be picked when a
        preceding condition is skipped (empty name). Also covers the case where
        the first condition ('M') always received dom_dur=None because the index
        counter was based on the result list rather than the input list.
        """
        # Condition at index 0 has an empty name and will be skipped.
        # Condition at index 1 ('HbO') should receive dom_durations[1] = 60.0,
        # NOT dom_durations[0] = None.
        state = _state(
            conditions=[
                _cond("", duration=1.0),   # index 0 — will be skipped (empty name)
                _cond("HbO", duration=1.0),  # index 1 — must get dom_durations[1]
            ]
        )
        store_data, is_open, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[None, 60.0],
            prev_gc=None,
        )
        assert is_open is False
        durations = {c["original_name"]: c["duration"] for c in store_data["conditions"]}
        assert durations["HbO"] == 60.0

    # A-16 --------------------------------------------------------------------
    def test_groups_preserved_in_apply(self) -> None:
        """Groups defined in state are not lost after Apply."""
        group = {
            "label": "All",
            "conditions": ["HbO"],
            "tmin": -2.0,
            "tmax": 18.0,
            "baseline_tmin": -2.0,
            "baseline_tmax": 0.0,
        }
        state = _state(
            conditions=[_cond("HbO", duration=5.0)],
            groups=[group],
        )
        store_data, is_open, *_ = apply_condition_config(
            n_clicks=1,
            state=state,
            dom_durations=[5.0],
            prev_gc=None,
        )
        assert is_open is False
        assert "groups" in store_data
        assert store_data["groups"][0]["label"] == "All"


# ---------------------------------------------------------------------------
# TestOpenConditionModalFromButton
# ---------------------------------------------------------------------------


class TestOpenConditionModalFromButton:
    """Tests for open_condition_modal_from_button callback."""

    # O-01 --------------------------------------------------------------------
    def test_restores_last_applied_duration_from_gc_store(self) -> None:
        """gc_store duration (8.0) is restored into the new state."""
        gc_store = _gc_store(conditions=[_gc_cond("HbO", duration=8.0)])
        state = _state(conditions=[_cond("HbO", duration=1.0)])
        _, new_state = open_condition_modal_from_button(
            n_clicks=1,
            state=state,
            global_conditions=gc_store,
        )
        assert new_state is not no_update
        restored = {c["original_name"]: c for c in new_state["conditions"]}
        assert restored["HbO"]["duration"] == 8.0

    # O-02 --------------------------------------------------------------------
    def test_restores_all_numeric_fields(self) -> None:
        """tmin, tmax, baseline_tmin, baseline_tmax restored from gc_store."""
        gc_store = _gc_store(
            conditions=[
                _gc_cond("HbO", duration=5.0, tmin=-3.0, tmax=20.0, baseline_tmin=-1.0, baseline_tmax=0.5)
            ]
        )
        state = _state(conditions=[_cond("HbO")])
        _, new_state = open_condition_modal_from_button(
            n_clicks=1, state=state, global_conditions=gc_store
        )
        c = new_state["conditions"][0]
        assert c["tmin"] == -3.0
        assert c["tmax"] == 20.0
        assert c["baseline_tmin"] == -1.0
        assert c["baseline_tmax"] == 0.5

    # O-03 --------------------------------------------------------------------
    def test_sets_open_true_in_state(self) -> None:
        """_open flag is set to True in returned state."""
        gc_store = _gc_store(conditions=[_gc_cond("HbO")])
        state = _state(conditions=[_cond("HbO")])
        _, new_state = open_condition_modal_from_button(
            n_clicks=1, state=state, global_conditions=gc_store
        )
        assert new_state.get("_open") is True

    # O-04 --------------------------------------------------------------------
    def test_no_update_when_n_clicks_none(self) -> None:
        """n_clicks=None → both outputs return no_update."""
        gc_store = _gc_store(conditions=[_gc_cond("HbO")])
        result = open_condition_modal_from_button(
            n_clicks=None, state=None, global_conditions=gc_store
        )
        assert all(r is no_update for r in result)

    # O-05 --------------------------------------------------------------------
    def test_no_update_when_gc_store_none(self) -> None:
        """global_conditions=None → both outputs return no_update."""
        result = open_condition_modal_from_button(
            n_clicks=1, state=None, global_conditions=None
        )
        assert all(r is no_update for r in result)

    # O-06 --------------------------------------------------------------------
    def test_gc_store_with_only_hbo_discards_hbr_from_state(self) -> None:
        """gc_store has only 'HbO' → returned state has only 'HbO' (HbR discarded by design)."""
        gc_store = _gc_store(conditions=[_gc_cond("HbO", duration=5.0)])
        state = _state(
            conditions=[
                _cond("HbO", duration=1.0),
                _cond("HbR", duration=2.0),
            ]
        )
        _, new_state = open_condition_modal_from_button(
            n_clicks=1, state=state, global_conditions=gc_store
        )
        names = [c["original_name"] for c in new_state["conditions"]]
        assert names == ["HbO"]
        assert "HbR" not in names

    # O-07 --------------------------------------------------------------------
    def test_groups_from_gc_store_are_restored(self) -> None:
        """Groups present in gc_store are written into new_state."""
        gc_groups = [
            {
                "label": "AllConds",
                "conditions": ["HbO"],
                "tmin": -2.0,
                "tmax": 18.0,
                "baseline_tmin": -2.0,
                "baseline_tmax": 0.0,
            }
        ]
        gc_store = _gc_store(
            conditions=[_gc_cond("HbO")],
            groups=gc_groups,
        )
        state = _state(conditions=[_cond("HbO")])
        _, new_state = open_condition_modal_from_button(
            n_clicks=1, state=state, global_conditions=gc_store
        )
        assert "groups" in new_state
        assert new_state["groups"][0]["label"] == "AllConds"

    # O-08 --------------------------------------------------------------------
    def test_no_groups_in_gc_store_preserves_state_groups(self) -> None:
        """gc_store has no 'groups' key → state groups are kept unchanged."""
        existing_groups = [
            {
                "label": "Existing",
                "conditions": ["HbO"],
                "tmin": -2.0,
                "tmax": 18.0,
                "baseline_tmin": -2.0,
                "baseline_tmax": 0.0,
            }
        ]
        gc_store = _gc_store(conditions=[_gc_cond("HbO")])  # no groups key
        state = _state(conditions=[_cond("HbO")], groups=existing_groups)
        _, new_state = open_condition_modal_from_button(
            n_clicks=1, state=state, global_conditions=gc_store
        )
        # gc_store had no groups → state groups must survive
        assert "groups" in new_state
        assert new_state["groups"][0]["label"] == "Existing"


# ---------------------------------------------------------------------------
# TestCancelConditionConfig
# ---------------------------------------------------------------------------


class TestCancelConditionConfig:
    """Tests for cancel_condition_config callback."""

    # C-01 --------------------------------------------------------------------
    def test_cancel_returns_false(self) -> None:
        """n_clicks=1 → is_open=False."""
        result = cancel_condition_config(n_clicks=1)
        assert result is False

    # C-02 --------------------------------------------------------------------
    def test_no_update_when_n_clicks_none(self) -> None:
        """n_clicks=None → no_update."""
        result = cancel_condition_config(n_clicks=None)
        assert result is no_update
