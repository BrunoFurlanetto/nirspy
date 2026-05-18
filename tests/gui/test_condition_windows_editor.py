"""Tests for the per-condition windows editor widget (T-012).

Conditions are SNIRF-driven only — there is no manual add/remove/rename
flow. These tests cover the two modes the editor exposes:

- Disabled mode: no available_conditions → switch disabled, instructional
  hint shown, no rows / no add button.
- Enabled mode: available_conditions list → one row per condition with
  the name displayed as a read-only label and four numeric inputs.
"""

from __future__ import annotations

from nirspy.gui.components.condition_windows_editor import (
    render_condition_windows_editor,
)


class TestConditionWindowsEditorDisabled:
    """Editor when no upstream SNIRF is reachable."""

    def test_switch_disabled(self) -> None:
        result = render_condition_windows_editor("inst-1", None)
        html_str = str(result)
        assert "cond-window-switch" in html_str
        assert "disabled=True" in html_str

    def test_hint_directs_user_to_loadsnirf(self) -> None:
        result = render_condition_windows_editor("inst-2", None)
        assert "Set the LoadSnirf path first" in str(result)

    def test_no_add_button(self) -> None:
        result = render_condition_windows_editor("inst-3", None)
        assert "cond-window-add" not in str(result)

    def test_no_rows_rendered(self) -> None:
        result = render_condition_windows_editor("inst-4", {})
        assert "cond-window-row" not in str(result)


class TestConditionWindowsEditorEnabled:
    """Editor when LoadSnirf supplies condition names."""

    def test_renders_one_row_per_condition(self) -> None:
        result = render_condition_windows_editor(
            "inst-5",
            {
                "A": {"tmin": -2.0, "tmax": 18.0,
                      "baseline_tmin": -2.0, "baseline_tmax": 0.0},
                "B": {"tmin": -2.0, "tmax": 18.0,
                      "baseline_tmin": -2.0, "baseline_tmax": 0.0},
            },
            available_conditions=["A", "B"],
        )
        html_str = str(result)
        assert "'condition': 'A'" in html_str
        assert "'condition': 'B'" in html_str

    def test_switch_off_hides_table_even_with_conditions(self) -> None:
        result = render_condition_windows_editor(
            "inst-6",
            {},
            available_conditions=["A", "B"],
        )
        assert "'display': 'none'" in str(result)

    def test_switch_on_shows_table(self) -> None:
        current = {
            "Rest": {"tmin": -2.0, "tmax": 18.0,
                     "baseline_tmin": -2.0, "baseline_tmax": 0.0},
        }
        result = render_condition_windows_editor(
            "inst-7", current, available_conditions=["Rest"]
        )
        assert "'display': 'block'" in str(result)

    def test_no_add_button_in_snirf_mode(self) -> None:
        result = render_condition_windows_editor(
            "inst-8",
            {"A": {"tmin": -2.0, "tmax": 18.0,
                   "baseline_tmin": -2.0, "baseline_tmax": 0.0}},
            available_conditions=["A"],
        )
        assert "cond-window-add" not in str(result)

    def test_no_remove_button_in_snirf_mode(self) -> None:
        result = render_condition_windows_editor(
            "inst-9",
            {"A": {"tmin": -2.0, "tmax": 18.0,
                   "baseline_tmin": -2.0, "baseline_tmax": 0.0}},
            available_conditions=["A"],
        )
        assert "cond-window-remove" not in str(result)

    def test_condition_name_not_in_input_value(self) -> None:
        """Name appears as a label, not as a text-input value."""
        result = render_condition_windows_editor(
            "inst-10",
            {"Tapping": {"tmin": -2.0, "tmax": 18.0,
                         "baseline_tmin": -2.0, "baseline_tmax": 0.0}},
            available_conditions=["Tapping"],
        )
        html_str = str(result)
        assert "Tapping" in html_str
        # No condition_name field in the row id any more
        assert "'field': 'condition_name'" not in html_str
