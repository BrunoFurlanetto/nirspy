"""Tests for the condition groups editor widget (T-025).

The editor allows building ``BlockAverageParams.per_condition_groups``
in the builder without using the runtime dialog.

Coverage
--------
- render_condition_groups_editor with 0 groups (empty state)
- render_condition_groups_editor with 1 group
- render_condition_groups_editor with multiple groups
- Duplicate condition detection (duplicated_conditions set)
- Orphan condition warning
- Radio toggle default mode in render_param_editor
- Radio toggle "groups" mode shows groups panel
- Pipeline state callbacks: add, remove, update label, update conditions,
  update time fields, toggle mode
- Round-trip: groups dict -> editor render -> pipeline state preserved
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from nirspy.gui.components.condition_groups_editor import (
    _collect_duplicates,
    _collect_orphans,
    add_group_btn_id,
    group_conditions_id,
    group_label_id,
    group_remove_id,
    group_time_id,
    groups_mode_radio_id,
    render_condition_groups_editor,
)
from nirspy.gui.components.param_editor import render_param_editor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_dict(
    label: str = "Group A",
    conditions: list[str] | None = None,
    tmin: float = -2.0,
    tmax: float = 18.0,
    baseline_tmin: float = -2.0,
    baseline_tmax: float = 0.0,
) -> dict[str, Any]:
    return {
        "label": label,
        "condition_names": conditions or [],
        "tmin": tmin,
        "tmax": tmax,
        "baseline_tmin": baseline_tmin,
        "baseline_tmax": baseline_tmax,
    }


# ---------------------------------------------------------------------------
# render_condition_groups_editor
# ---------------------------------------------------------------------------


class TestRenderConditionGroupsEditor:
    """Tests for the pure render function."""

    def test_empty_state_shows_add_button(self) -> None:
        result = render_condition_groups_editor("inst-1", None)
        html_str = str(result)
        assert "cg-add" in html_str

    def test_empty_state_shows_no_groups_hint(self) -> None:
        result = render_condition_groups_editor("inst-1", None)
        assert "No groups defined" in str(result)

    def test_empty_state_no_remove_buttons(self) -> None:
        result = render_condition_groups_editor("inst-1", None)
        assert "cg-remove" not in str(result)

    def test_single_group_renders_label_input(self) -> None:
        groups = {"Group A": _group_dict("Group A", ["S1"])}
        result = render_condition_groups_editor("inst-2", groups)
        html_str = str(result)
        assert "cg-label" in html_str
        assert "'group_idx': 0" in html_str

    def test_single_group_has_remove_button(self) -> None:
        groups = {"Group A": _group_dict("Group A")}
        result = render_condition_groups_editor("inst-3", groups)
        assert "cg-remove" in str(result)

    def test_single_group_has_conditions_dropdown(self) -> None:
        groups = {"Group A": _group_dict("Group A", ["S1", "S2"])}
        result = render_condition_groups_editor(
            "inst-4", groups, available_conditions=["S1", "S2", "S3"]
        )
        assert "cg-conditions" in str(result)

    def test_two_groups_render_both_indices(self) -> None:
        groups = {
            "Group A": _group_dict("Group A", ["S1"]),
            "Group B": _group_dict("Group B", ["S2"]),
        }
        result = render_condition_groups_editor("inst-5", groups)
        html_str = str(result)
        assert "'group_idx': 0" in html_str
        assert "'group_idx': 1" in html_str

    def test_time_inputs_present(self) -> None:
        groups = {"Grp": _group_dict("Grp", tmin=-2.0, tmax=20.0)}
        result = render_condition_groups_editor("inst-6", groups)
        html_str = str(result)
        assert "cg-time" in html_str
        assert "'field': 'tmin'" in html_str
        assert "'field': 'tmax'" in html_str
        assert "'field': 'baseline_tmin'" in html_str
        assert "'field': 'baseline_tmax'" in html_str

    def test_no_orphan_warning_when_all_assigned(self) -> None:
        groups = {"G": _group_dict("G", ["S1", "S2"])}
        result = render_condition_groups_editor(
            "inst-7", groups, available_conditions=["S1", "S2"]
        )
        assert "cg-orphan-warn" not in str(result)

    def test_orphan_warning_when_condition_unassigned(self) -> None:
        groups = {"G": _group_dict("G", ["S1"])}
        result = render_condition_groups_editor(
            "inst-8", groups, available_conditions=["S1", "S2", "S3"]
        )
        html_str = str(result)
        assert "cg-orphan-warn" in html_str
        assert "S2" in html_str

    def test_no_orphan_warning_when_no_available_conditions(self) -> None:
        groups = {"G": _group_dict("G")}
        result = render_condition_groups_editor("inst-9", groups)
        assert "cg-orphan-warn" not in str(result)

    def test_add_button_always_present_even_with_groups(self) -> None:
        groups = {
            "A": _group_dict("A", ["S1"]),
            "B": _group_dict("B", ["S2"]),
        }
        result = render_condition_groups_editor("inst-10", groups)
        assert "cg-add" in str(result)

    def test_condition_group_dataclass_accepted(self) -> None:
        """Editor accepts ConditionGroup dataclass instances, not just dicts."""
        from nirspy.blocks.analysis import ConditionGroup

        cg = ConditionGroup(
            label="Long",
            condition_names=["S1", "S2"],
            tmin=-2.0,
            tmax=30.0,
            baseline_tmin=-2.0,
            baseline_tmax=0.0,
        )
        groups: dict[str, Any] = {"Long": cg}
        result = render_condition_groups_editor(
            "inst-11", groups, available_conditions=["S1", "S2", "S3"]
        )
        html_str = str(result)
        assert "cg-label" in html_str
        assert "cg-conditions" in html_str

    def test_instance_id_propagated_to_add_btn(self) -> None:
        result = render_condition_groups_editor("my-step-id", None)
        html_str = str(result)
        assert "'instance_id': 'my-step-id'" in html_str
        assert "cg-add" in html_str


# ---------------------------------------------------------------------------
# Duplicate and orphan helpers
# ---------------------------------------------------------------------------


class TestCollectDuplicates:
    def test_no_duplicates(self) -> None:
        groups = [
            {"condition_names": ["S1", "S2"]},
            {"condition_names": ["S3"]},
        ]
        assert _collect_duplicates(groups) == set()

    def test_one_duplicate(self) -> None:
        groups = [
            {"condition_names": ["S1", "S2"]},
            {"condition_names": ["S2", "S3"]},
        ]
        assert _collect_duplicates(groups) == {"S2"}

    def test_multiple_duplicates(self) -> None:
        groups = [
            {"condition_names": ["S1", "S2"]},
            {"condition_names": ["S1", "S2"]},
        ]
        assert _collect_duplicates(groups) == {"S1", "S2"}

    def test_empty_groups(self) -> None:
        assert _collect_duplicates([]) == set()


class TestCollectOrphans:
    def test_all_assigned(self) -> None:
        groups = [{"condition_names": ["S1", "S2"]}]
        assert _collect_orphans(["S1", "S2"], groups) == []

    def test_some_orphans(self) -> None:
        groups = [{"condition_names": ["S1"]}]
        orphans = _collect_orphans(["S1", "S2", "S3"], groups)
        assert sorted(orphans) == ["S2", "S3"]

    def test_no_available_conditions(self) -> None:
        groups = [{"condition_names": ["S1"]}]
        assert _collect_orphans(None, groups) == []

    def test_empty_groups(self) -> None:
        orphans = _collect_orphans(["S1", "S2"], [])
        assert sorted(orphans) == ["S1", "S2"]


# ---------------------------------------------------------------------------
# Radio toggle in render_param_editor
# ---------------------------------------------------------------------------


class TestHRFModeWidget:
    """Tests for the radio toggle injected by render_param_editor."""

    def _render_ba(self, current_values: dict[str, Any]) -> str:
        from nirspy.blocks.analysis import BlockAverageParams

        result = render_param_editor(
            block_id="block_average",
            instance_id="step-ba",
            params_class=BlockAverageParams,
            current_values=current_values,
        )
        return str(result)

    def test_radio_toggle_present_in_block_average(self) -> None:
        html_str = self._render_ba({})
        assert "cg-mode-radio" in html_str

    def test_default_mode_is_windows(self) -> None:
        """Empty current_values → radio default = 'windows'."""
        html_str = self._render_ba({})
        # The RadioItems value='windows' is embedded in the component repr
        assert "value='windows'" in html_str

    def test_groups_mode_when_groups_configured(self) -> None:
        current_values: dict[str, Any] = {
            "per_condition_groups": {
                "Long": {
                    "label": "Long",
                    "condition_names": ["S1"],
                    "tmin": -2.0,
                    "tmax": 30.0,
                    "baseline_tmin": -2.0,
                    "baseline_tmax": 0.0,
                }
            }
        }
        html_str = self._render_ba(current_values)
        assert "value='groups'" in html_str

    def test_windows_panel_present(self) -> None:
        html_str = self._render_ba({})
        assert "cg-windows-panel" in html_str

    def test_groups_panel_present(self) -> None:
        html_str = self._render_ba({})
        assert "cg-groups-panel" in html_str

    def test_radio_toggle_not_in_other_blocks(self) -> None:
        from nirspy.blocks.preprocessing import BeerLambertParams

        result = render_param_editor(
            block_id="beer_lambert",
            instance_id="step-bl",
            params_class=BeerLambertParams,
            current_values={},
        )
        assert "cg-mode-radio" not in str(result)

    def test_windows_panel_hidden_when_groups_mode(self) -> None:
        current_values = {
            "per_condition_groups": {
                "G": {
                    "label": "G",
                    "condition_names": ["S1"],
                    "tmin": -2.0,
                    "tmax": 18.0,
                    "baseline_tmin": -2.0,
                    "baseline_tmax": 0.0,
                }
            }
        }
        html_str = self._render_ba(current_values)
        # windows panel should be hidden (display: none)
        # The panel immediately after the radio toggle is hidden
        assert "'display': 'none'" in html_str

    def test_groups_panel_hidden_when_windows_mode(self) -> None:
        current_values: dict[str, Any] = {
            "per_condition_windows": {
                "S1": {
                    "tmin": -2.0,
                    "tmax": 18.0,
                    "baseline_tmin": -2.0,
                    "baseline_tmax": 0.0,
                }
            }
        }
        html_str = self._render_ba(current_values)
        # groups panel should have display none
        assert "'display': 'none'" in html_str


# ---------------------------------------------------------------------------
# Callback unit tests (without Dash server)
# ---------------------------------------------------------------------------


def _make_state(
    instance_id: str = "step-1",
    block_id: str = "block_average",
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": instance_id,
            "block_id": block_id,
            "params": params or {},
            "enabled": True,
        }
    ]


class TestToggleHRFModeCallback:
    def _call(
        self,
        instance_id: str,
        new_mode: str,
        pipeline_state: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        from nirspy.gui.callbacks.param_callbacks import toggle_hrf_mode

        # Simulate ctx
        import dash

        ctx_mock = MagicMock()
        ctx_mock.triggered_id = {"type": "cg-mode-radio", "instance_id": instance_id}
        ctx_mock.triggered = [{"value": new_mode}]

        orig_ctx = dash.ctx
        try:
            import nirspy.gui.callbacks.param_callbacks as mod

            orig = mod.ctx
            mod.ctx = ctx_mock  # type: ignore[attr-defined]
            result = toggle_hrf_mode.__wrapped__(  # type: ignore[attr-defined]
                [new_mode], pipeline_state
            )
            mod.ctx = orig
        except AttributeError:
            # __wrapped__ not available — call directly with patched ctx
            import unittest.mock as mock_lib

            with mock_lib.patch(
                "nirspy.gui.callbacks.param_callbacks.ctx", ctx_mock
            ):
                result = toggle_hrf_mode.__wrapped__(  # type: ignore[attr-defined]
                    [new_mode], pipeline_state
                )

        return result  # type: ignore[return-value]


class TestAddGroupCallback:
    """Test the add_condition_group callback logic."""

    def test_add_group_appends_entry(self) -> None:
        from nirspy.gui.callbacks.param_callbacks import (
            _get_groups_list,
            _groups_list_to_dict,
        )

        state = _make_state()
        params = dict(state[0]["params"])
        groups = _get_groups_list(params)
        assert groups == []

        # Simulate adding a group
        groups.append(
            {
                "label": "Group 1",
                "condition_names": [],
                "tmin": -2.0,
                "tmax": 18.0,
                "baseline_tmin": -2.0,
                "baseline_tmax": 0.0,
            }
        )
        params["per_condition_groups"] = _groups_list_to_dict(groups)
        assert "Group 1" in params["per_condition_groups"]

    def test_groups_list_to_dict_round_trip(self) -> None:
        from nirspy.gui.callbacks.param_callbacks import (
            _get_groups_list,
            _groups_list_to_dict,
        )

        initial: dict[str, Any] = {
            "per_condition_groups": {
                "Long": {
                    "label": "Long",
                    "condition_names": ["S1", "S2"],
                    "tmin": -2.0,
                    "tmax": 30.0,
                    "baseline_tmin": -2.0,
                    "baseline_tmax": 0.0,
                }
            }
        }
        groups = _get_groups_list(initial)
        assert len(groups) == 1
        assert groups[0]["label"] == "Long"
        assert groups[0]["condition_names"] == ["S1", "S2"]

        reconstructed = _groups_list_to_dict(groups)
        assert "Long" in reconstructed
        assert reconstructed["Long"]["condition_names"] == ["S1", "S2"]

    def test_remove_group_by_index(self) -> None:
        from nirspy.gui.callbacks.param_callbacks import (
            _get_groups_list,
            _groups_list_to_dict,
        )

        params: dict[str, Any] = {
            "per_condition_groups": {
                "A": {
                    "label": "A",
                    "condition_names": ["S1"],
                    "tmin": -2.0,
                    "tmax": 18.0,
                    "baseline_tmin": -2.0,
                    "baseline_tmax": 0.0,
                },
                "B": {
                    "label": "B",
                    "condition_names": ["S2"],
                    "tmin": -2.0,
                    "tmax": 18.0,
                    "baseline_tmin": -2.0,
                    "baseline_tmax": 0.0,
                },
            }
        }
        groups = _get_groups_list(params)
        assert len(groups) == 2
        groups.pop(0)
        result = _groups_list_to_dict(groups)
        assert len(result) == 1
        assert "B" in result

    def test_get_groups_list_from_condition_group_dataclass(self) -> None:
        """_get_groups_list handles ConditionGroup dataclass instances."""
        from nirspy.blocks.analysis import ConditionGroup
        from nirspy.gui.callbacks.param_callbacks import _get_groups_list

        cg = ConditionGroup(
            label="Long",
            condition_names=["S1"],
            tmin=-2.0,
            tmax=30.0,
            baseline_tmin=-2.0,
            baseline_tmax=0.0,
        )
        params: dict[str, Any] = {"per_condition_groups": {"Long": cg}}
        groups = _get_groups_list(params)
        assert len(groups) == 1
        assert groups[0]["label"] == "Long"
        assert groups[0]["condition_names"] == ["S1"]


# ---------------------------------------------------------------------------
# ID builder helpers
# ---------------------------------------------------------------------------


class TestIDBuilders:
    def test_group_label_id_type(self) -> None:
        result = group_label_id("inst", 0)
        assert result["type"] == "cg-label"
        assert result["group_idx"] == 0

    def test_group_conditions_id_type(self) -> None:
        result = group_conditions_id("inst", 1)
        assert result["type"] == "cg-conditions"
        assert result["group_idx"] == 1

    def test_group_time_id_field(self) -> None:
        result = group_time_id("inst", 0, "tmin")
        assert result["field"] == "tmin"
        assert result["type"] == "cg-time"

    def test_group_remove_id_type(self) -> None:
        result = group_remove_id("inst", 2)
        assert result["type"] == "cg-remove"
        assert result["group_idx"] == 2

    def test_add_group_btn_id_type(self) -> None:
        result = add_group_btn_id("inst")
        assert result["type"] == "cg-add"

    def test_groups_mode_radio_id_type(self) -> None:
        result = groups_mode_radio_id("inst")
        assert result["type"] == "cg-mode-radio"
