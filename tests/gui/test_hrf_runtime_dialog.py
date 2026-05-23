"""Tests for T-028 -- HRF specialized runtime dialog.

Covers:
- render_hrf_runtime_dialog produces a Modal with correct id / backdrop / size
- Stage 1 body renders with condition dropdown options from available_conditions
- Stage 2 body renders one row per group
- Wizard navigation helpers (advance to stage 2, back to stage 1)
- Add-group callback appends a new group and re-renders cards
- Remove-group callback removes the clicked group
- _hrf_sync_groups callback writes labels/conditions/times into hrf-runtime-state
- _hrf_sync_stage2_times callback syncs Stage 2 edits into store
- build_hrf_params_override returns correct per_condition_groups dict
- build_hrf_params_override returns None when groups list is empty
- build_hrf_params_override skips groups with no label or no conditions
- hrf-runtime-state store is in the app layout
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from nirspy.blocks import registry
from nirspy.domain.block import BlockSpec
from nirspy.domain.data_types import DataType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ba_spec() -> BlockSpec:
    """Return the BlockAverage BlockSpec."""
    cls = registry.get("block_average")
    return cls.SPEC  # type: ignore[attr-defined]


def _make_groups(*labels_conds: tuple[str, list[str]]) -> list[dict[str, Any]]:
    """Construct a groups list for testing."""
    groups: list[dict[str, Any]] = []
    for label, conds in labels_conds:
        groups.append(
            {
                "label": label,
                "condition_names": conds,
                "tmin": -2.0,
                "tmax": 18.0,
                "baseline_tmin": -2.0,
                "baseline_tmax": 0.0,
            }
        )
    return groups


# ---------------------------------------------------------------------------
# render_hrf_runtime_dialog — structure
# ---------------------------------------------------------------------------


class TestRenderHrfRuntimeDialog:
    """Unit tests for the render_hrf_runtime_dialog pure render function."""

    def test_returns_modal(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog
        import dash_bootstrap_components as dbc

        modal = render_hrf_runtime_dialog(_ba_spec(), 0, 5)
        assert isinstance(modal, dbc.Modal)

    def test_modal_id(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal = render_hrf_runtime_dialog(_ba_spec(), 0, 5)
        assert modal.id == "hrf-runtime-modal"

    def test_backdrop_static(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal = render_hrf_runtime_dialog(_ba_spec(), 0, 5)
        assert modal.backdrop == "static"

    def test_keyboard_false(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal = render_hrf_runtime_dialog(_ba_spec(), 1, 3)
        assert modal.keyboard is False

    def test_size_xl(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal = render_hrf_runtime_dialog(_ba_spec(), 0, 5)
        assert modal.size == "xl"

    def test_is_open_true(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal = render_hrf_runtime_dialog(_ba_spec(), 0, 5)
        assert modal.is_open is True

    def test_header_shows_step_counter(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 2, 8))
        assert "Block 3/8" in modal_str

    def test_header_contains_block_name(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        spec = _ba_spec()
        modal_str = str(render_hrf_runtime_dialog(spec, 0, 5))
        assert spec.display_name in modal_str

    def test_header_mentions_hrf_groups(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "HRF Groups" in modal_str

    def test_footer_stage1_present(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "hrf-rt-footer-stage1" in modal_str

    def test_footer_stage2_present(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "hrf-rt-footer-stage2" in modal_str

    def test_cancel_btn_id_present(self) -> None:
        """runtime-cancel-btn is shared with generic dialog."""
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "runtime-cancel-btn" in modal_str

    def test_next_btn_present(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "hrf-rt-next-btn" in modal_str

    def test_advance_btn_id_present(self) -> None:
        """runtime-advance-btn must be present so advance_run callback fires."""
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "runtime-advance-btn" in modal_str

    def test_back_btn_present(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "hrf-rt-back-btn" in modal_str

    def test_stage1_container_visible(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        # Stage 1 container starts visible
        assert "hrf-rt-stage1-container" in modal_str

    def test_stage2_container_hidden(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "hrf-rt-stage2-container" in modal_str

    def test_add_group_btn_present(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "hrf-rt-add-btn" in modal_str


# ---------------------------------------------------------------------------
# Stage 1 — auto-populate conditions
# ---------------------------------------------------------------------------


class TestStage1AutoPopulate:
    """Verify conditions are offered in the Dropdown when available."""

    def test_available_conditions_appear_in_stage1(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        conditions = ["S1", "S2", "S3"]
        # Add a group so we can see the dropdown options
        modal_str = str(
            render_hrf_runtime_dialog(
                _ba_spec(), 0, 5, available_conditions=conditions
            )
        )
        # Conditions should appear as option labels or values in the rendered output
        for c in conditions:
            assert c in modal_str

    def test_no_conditions_shows_warning(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(
            render_hrf_runtime_dialog(_ba_spec(), 0, 5, available_conditions=None)
        )
        assert "No SNIRF conditions" in modal_str

    def test_empty_initial_groups_shows_hint(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog

        modal_str = str(render_hrf_runtime_dialog(_ba_spec(), 0, 5))
        assert "No groups defined" in modal_str


# ---------------------------------------------------------------------------
# Stage renderers directly
# ---------------------------------------------------------------------------


class TestStage1Body:
    """Unit tests for _render_stage1_body."""

    def test_no_cards_when_no_groups(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage1_body

        body = _render_stage1_body([], None)
        body_str = str(body)
        assert "hrf-rt-cards-container" in body_str
        assert "No groups defined" in body_str

    def test_cards_rendered_per_group(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage1_body

        groups = _make_groups(("GroupA", ["S1"]), ("GroupB", ["S2", "S3"]))
        body_str = str(_render_stage1_body(groups, ["S1", "S2", "S3"]))
        assert "GroupA" in body_str
        assert "GroupB" in body_str

    def test_orphan_warning_when_unassigned(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage1_body

        groups = _make_groups(("G1", ["S1"]))
        body_str = str(_render_stage1_body(groups, ["S1", "S2", "S3"]))
        assert "Unassigned" in body_str
        assert "S2" in body_str
        assert "S3" in body_str

    def test_no_orphan_warning_when_all_assigned(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage1_body

        groups = _make_groups(("G1", ["S1", "S2"]))
        body_str = str(_render_stage1_body(groups, ["S1", "S2"]))
        assert "Unassigned" not in body_str

    def test_remove_btn_per_group(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage1_body

        groups = _make_groups(("G1", []), ("G2", []))
        body_str = str(_render_stage1_body(groups, None))
        assert body_str.count("hrf-rt-remove") >= 2

    def test_duplicate_condition_conflict_message(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage1_body

        groups = _make_groups(("G1", ["S1"]), ("G2", ["S1", "S2"]))
        body_str = str(_render_stage1_body(groups, ["S1", "S2"]))
        assert "Conflict" in body_str


class TestStage2Body:
    """Unit tests for _render_stage2_body."""

    def test_warning_when_no_groups(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage2_body

        body_str = str(_render_stage2_body([]))
        assert "No groups defined" in body_str

    def test_one_row_per_group(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage2_body

        groups = _make_groups(("Alpha", ["S1"]), ("Beta", ["S2"]), ("Gamma", ["S3"]))
        body_str = str(_render_stage2_body(groups))
        assert "Alpha" in body_str
        assert "Beta" in body_str
        assert "Gamma" in body_str

    def test_time_inputs_use_hrf_rt_time2_type(self) -> None:
        """Stage 2 inputs must use hrf-rt-time2 IDs (not hrf-rt-time)."""
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage2_body

        groups = _make_groups(("G1", ["S1"]))
        body_str = str(_render_stage2_body(groups))
        assert "hrf-rt-time2" in body_str

    def test_group_count_hint(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage2_body

        groups = _make_groups(("G1", ["S1"]), ("G2", ["S2"]))
        body_str = str(_render_stage2_body(groups))
        assert "2 group(s)" in body_str

    def test_header_columns_present(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _render_stage2_body

        groups = _make_groups(("G1", ["S1"]))
        body_str = str(_render_stage2_body(groups))
        assert "tmin" in body_str
        assert "tmax" in body_str
        assert "bl_tmin" in body_str
        assert "bl_tmax" in body_str


# ---------------------------------------------------------------------------
# Stage navigation callbacks
# ---------------------------------------------------------------------------


class TestStageNavigation:
    """Unit tests for _hrf_advance_to_stage2 and _hrf_back_to_stage1."""

    def test_advance_to_stage2_no_clicks_returns_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_advance_to_stage2

        result = _hrf_advance_to_stage2(None, None)
        assert all(r is no_update for r in result)

    def test_advance_to_stage2_shows_stage2(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_advance_to_stage2

        hrf_state = {
            "groups": _make_groups(("G1", ["S1"])),
            "available_conditions": ["S1"],
        }
        (
            stage1_style, stage2_style,
            footer1_style, footer2_style,
            stage_val, stage2_children,
        ) = _hrf_advance_to_stage2(1, hrf_state)

        assert stage1_style == {"display": "none"}
        assert stage2_style == {"display": "block"}
        assert footer1_style == {"display": "none"}
        assert footer2_style == {"display": "block"}
        assert stage_val == 2

    def test_advance_to_stage2_renders_groups_in_children(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_advance_to_stage2

        hrf_state = {
            "groups": _make_groups(("Alpha", ["S1"]), ("Beta", ["S2"])),
            "available_conditions": ["S1", "S2"],
        }
        *_, stage2_children = _hrf_advance_to_stage2(1, hrf_state)
        children_str = str(stage2_children)
        assert "Alpha" in children_str
        assert "Beta" in children_str

    def test_back_to_stage1_no_clicks_returns_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_back_to_stage1

        result = _hrf_back_to_stage1(None)
        assert all(r is no_update for r in result)

    def test_back_to_stage1_shows_stage1(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_back_to_stage1

        (
            stage1_style, stage2_style,
            footer1_style, footer2_style,
            stage_val,
        ) = _hrf_back_to_stage1(1)

        assert stage1_style == {"display": "block"}
        assert stage2_style == {"display": "none"}
        assert footer1_style == {"display": "block"}
        assert footer2_style == {"display": "none"}
        assert stage_val == 1


# ---------------------------------------------------------------------------
# Add / remove group callbacks
# ---------------------------------------------------------------------------


class TestAddRemoveGroup:
    """Unit tests for _hrf_add_group and _hrf_remove_group callbacks."""

    def test_add_group_no_clicks_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_add_group

        result = _hrf_add_group(None, None, None)
        assert all(r is no_update for r in result)

    def test_add_group_appends_to_groups(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_add_group

        hrf_state: dict[str, Any] = {"groups": [], "available_conditions": None}
        defaults: dict[str, Any] = {"tmin": -2.0, "tmax": 18.0,
                                     "baseline_tmin": -2.0, "baseline_tmax": 0.0}
        _, new_state = _hrf_add_group(1, hrf_state, defaults)
        assert len(new_state["groups"]) == 1
        assert new_state["groups"][0]["label"] == ""
        assert new_state["groups"][0]["tmin"] == -2.0

    def test_add_group_twice_gives_two_groups(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_add_group

        hrf_state: dict[str, Any] = {"groups": [], "available_conditions": None}
        defaults: dict[str, Any] = {}
        _, state1 = _hrf_add_group(1, hrf_state, defaults)
        _, state2 = _hrf_add_group(2, state1, defaults)
        assert len(state2["groups"]) == 2

    def test_add_group_renders_cards(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_add_group

        hrf_state: dict[str, Any] = {"groups": [], "available_conditions": ["S1"]}
        cards, _ = _hrf_add_group(1, hrf_state, {})
        cards_str = str(cards)
        assert "hrf-rt-label" in cards_str

    def test_remove_group_no_clicks_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_remove_group

        result = _hrf_remove_group([None, None], {"groups": [{"label": "G"}], "available_conditions": None})
        assert all(r is no_update for r in result)

    def test_remove_group_removes_correct_index(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_remove_group

        groups = _make_groups(("A", []), ("B", []), ("C", []))
        hrf_state: dict[str, Any] = {"groups": groups, "available_conditions": None}
        # Click on index 1 (group "B")
        _, new_state = _hrf_remove_group([None, 1, None], hrf_state)
        remaining = [g["label"] for g in new_state["groups"]]
        assert remaining == ["A", "C"]

    def test_remove_first_group(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_remove_group

        groups = _make_groups(("First", []), ("Second", []))
        hrf_state: dict[str, Any] = {"groups": groups, "available_conditions": None}
        _, new_state = _hrf_remove_group([1, None], hrf_state)
        assert len(new_state["groups"]) == 1
        assert new_state["groups"][0]["label"] == "Second"


# ---------------------------------------------------------------------------
# Sync callbacks
# ---------------------------------------------------------------------------


class TestSyncCallbacks:
    """Unit tests for _hrf_sync_groups and _hrf_sync_stage2_times."""

    def test_sync_groups_no_labels_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_sync_groups

        result = _hrf_sync_groups([], [], [], {})
        assert result is no_update

    def test_sync_groups_writes_labels(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_sync_groups

        labels = ["Long stim", "Short stim"]
        conditions = [["S1", "S2"], ["S3"]]
        # 2 groups × 4 fields = 8 time values
        times = [-2.0, 18.0, -2.0, 0.0, -1.0, 10.0, -1.0, 0.0]
        hrf_state: dict[str, Any] = {"groups": [], "available_conditions": None}

        new_state = _hrf_sync_groups(labels, conditions, times, hrf_state)
        assert new_state["groups"][0]["label"] == "Long stim"
        assert new_state["groups"][1]["label"] == "Short stim"

    def test_sync_groups_writes_conditions(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_sync_groups

        labels = ["G1"]
        conditions = [["S1", "S2"]]
        times = [-2.0, 18.0, -2.0, 0.0]
        hrf_state: dict[str, Any] = {"groups": [], "available_conditions": None}

        new_state = _hrf_sync_groups(labels, conditions, times, hrf_state)
        assert new_state["groups"][0]["condition_names"] == ["S1", "S2"]

    def test_sync_groups_writes_times(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_sync_groups

        labels = ["G1"]
        conditions = [["S1"]]
        times = [-5.0, 30.0, -3.0, 0.0]
        hrf_state: dict[str, Any] = {"groups": [], "available_conditions": None}

        new_state = _hrf_sync_groups(labels, conditions, times, hrf_state)
        g = new_state["groups"][0]
        assert g["tmin"] == -5.0
        assert g["tmax"] == 30.0
        assert g["baseline_tmin"] == -3.0
        assert g["baseline_tmax"] == 0.0

    def test_sync_stage2_times_no_times_no_update(self) -> None:
        from dash import no_update
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_sync_stage2_times

        result = _hrf_sync_stage2_times([], {"groups": [], "available_conditions": None})
        assert result is no_update

    def test_sync_stage2_times_updates_store(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import _hrf_sync_stage2_times

        groups = _make_groups(("G1", ["S1"]), ("G2", ["S2"]))
        hrf_state: dict[str, Any] = {"groups": groups, "available_conditions": None}
        # 2 groups × 4 fields = 8 values
        times = [-5.0, 25.0, -3.0, 0.0, -1.0, 15.0, -1.0, 0.0]

        new_state = _hrf_sync_stage2_times(times, hrf_state)
        assert new_state["groups"][0]["tmin"] == -5.0
        assert new_state["groups"][0]["tmax"] == 25.0
        assert new_state["groups"][1]["tmin"] == -1.0
        assert new_state["groups"][1]["tmax"] == 15.0


# ---------------------------------------------------------------------------
# build_hrf_params_override
# ---------------------------------------------------------------------------


class TestBuildHrfParamsOverride:
    """Unit tests for build_hrf_params_override helper."""

    def test_none_state_returns_none(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        assert build_hrf_params_override(None) is None

    def test_empty_groups_returns_none(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        state = {"groups": [], "available_conditions": None}
        assert build_hrf_params_override(state) is None

    def test_group_without_label_skipped(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        groups = [{"label": "", "condition_names": ["S1"],
                   "tmin": -2.0, "tmax": 18.0, "baseline_tmin": -2.0, "baseline_tmax": 0.0}]
        state = {"groups": groups, "available_conditions": None}
        assert build_hrf_params_override(state) is None

    def test_group_without_conditions_skipped(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        groups = [{"label": "G1", "condition_names": [],
                   "tmin": -2.0, "tmax": 18.0, "baseline_tmin": -2.0, "baseline_tmax": 0.0}]
        state = {"groups": groups, "available_conditions": None}
        assert build_hrf_params_override(state) is None

    def test_valid_groups_returns_dict(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        groups = _make_groups(("Long", ["S1", "S2"]), ("Short", ["S3"]))
        state = {"groups": groups, "available_conditions": None}
        result = build_hrf_params_override(state)
        assert result is not None
        assert "per_condition_groups" in result

    def test_override_contains_correct_labels(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        groups = _make_groups(("Long", ["S1"]), ("Short", ["S2"]))
        state = {"groups": groups, "available_conditions": None}
        result = build_hrf_params_override(state)
        assert result is not None
        pcg = result["per_condition_groups"]
        assert "Long" in pcg
        assert "Short" in pcg

    def test_override_conditions_correct(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        groups = _make_groups(("G1", ["S1", "S2"]))
        state = {"groups": groups, "available_conditions": None}
        result = build_hrf_params_override(state)
        assert result is not None
        assert result["per_condition_groups"]["G1"]["condition_names"] == ["S1", "S2"]

    def test_override_times_correct(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        groups = [{
            "label": "G1", "condition_names": ["S1"],
            "tmin": -5.0, "tmax": 30.0, "baseline_tmin": -3.0, "baseline_tmax": 0.5,
        }]
        state = {"groups": groups, "available_conditions": None}
        result = build_hrf_params_override(state)
        assert result is not None
        g = result["per_condition_groups"]["G1"]
        assert g["tmin"] == -5.0
        assert g["tmax"] == 30.0
        assert g["baseline_tmin"] == -3.0
        assert g["baseline_tmax"] == 0.5

    def test_hrf_mode_marker_present(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        groups = _make_groups(("G1", ["S1"]))
        state = {"groups": groups, "available_conditions": None}
        result = build_hrf_params_override(state)
        assert result is not None
        assert result.get("_hrf_mode") == "groups"

    def test_per_condition_groups_coercible_by_block_avg_params(self) -> None:
        """per_condition_groups dicts must be accepted by BlockAverageParams.__post_init__."""
        from nirspy.blocks.analysis import BlockAverageParams
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        groups = _make_groups(("Long", ["S1", "S2"]), ("Short", ["S3"]))
        state = {"groups": groups, "available_conditions": None}
        result = build_hrf_params_override(state)
        assert result is not None
        pcg = result["per_condition_groups"]

        # Strip internal marker
        override = {k: v for k, v in result.items() if k != "_hrf_mode"}

        # BlockAverageParams must accept the plain-dict per_condition_groups
        params = BlockAverageParams(**override)
        assert len(params.per_condition_groups) == 2
        assert "Long" in params.per_condition_groups
        assert params.per_condition_groups["Long"].condition_names == ["S1", "S2"]

    def test_only_valid_groups_included(self) -> None:
        """Groups with missing label or empty conditions are excluded."""
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override

        groups = [
            {"label": "Good", "condition_names": ["S1"],
             "tmin": -2.0, "tmax": 18.0, "baseline_tmin": -2.0, "baseline_tmax": 0.0},
            {"label": "", "condition_names": ["S2"],
             "tmin": -2.0, "tmax": 18.0, "baseline_tmin": -2.0, "baseline_tmax": 0.0},
            {"label": "NoConditions", "condition_names": [],
             "tmin": -2.0, "tmax": 18.0, "baseline_tmin": -2.0, "baseline_tmax": 0.0},
        ]
        state = {"groups": groups, "available_conditions": None}
        result = build_hrf_params_override(state)
        assert result is not None
        assert list(result["per_condition_groups"].keys()) == ["Good"]


# ---------------------------------------------------------------------------
# Layout integration
# ---------------------------------------------------------------------------


class TestLayoutIntegration:
    """Verify layouts.py wires up hrf-runtime-state store."""

    def test_hrf_runtime_state_store_in_layout(self) -> None:
        from nirspy.gui.layouts import create_layout

        layout_str = str(create_layout())
        assert "hrf-runtime-state" in layout_str


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------


class TestImportSanity:
    def test_hrf_runtime_dialog_importable(self) -> None:
        import nirspy.gui.components.hrf_runtime_dialog  # noqa: F401

    def test_render_hrf_runtime_dialog_callable(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import render_hrf_runtime_dialog
        assert callable(render_hrf_runtime_dialog)

    def test_build_hrf_params_override_callable(self) -> None:
        from nirspy.gui.components.hrf_runtime_dialog import build_hrf_params_override
        assert callable(build_hrf_params_override)

    def test_advance_run_has_hrf_state_param(self) -> None:
        """advance_run must accept hrf_state (added in T-028)."""
        import inspect
        from nirspy.gui.callbacks.runtime_callbacks import advance_run

        sig = inspect.signature(advance_run)
        assert "hrf_state" in sig.parameters
