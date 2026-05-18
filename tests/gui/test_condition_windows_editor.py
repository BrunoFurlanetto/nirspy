"""Tests for the per-condition windows editor widget (T-012)."""

from __future__ import annotations

from nirspy.gui.components.condition_windows_editor import (
    render_condition_windows_editor,
)


class TestConditionWindowsEditor:
    """Tests for render_condition_windows_editor."""

    def test_render_empty(self) -> None:
        result = render_condition_windows_editor("inst-1", None)
        html_str = str(result)
        assert "Per-condition windows" in html_str
        assert "cond-window-switch" in html_str
        assert "none" in html_str  # table hidden

    def test_render_with_values(self) -> None:
        current = {
            "Tapping": {
                "tmin": -5.0,
                "tmax": 30.0,
                "baseline_tmin": -5.0,
                "baseline_tmax": 0.0,
            }
        }
        result = render_condition_windows_editor("inst-2", current)
        html_str = str(result)
        assert "Tapping" in html_str
        assert "block" in html_str  # table visible (display: block)

    def test_render_with_available_conditions(self) -> None:
        result = render_condition_windows_editor(
            "inst-3",
            {},
            available_conditions=["A", "B", "C"],
        )
        html_str = str(result)
        assert "cond-window-switch" in html_str
        # No hint when conditions are available
        assert "Run pipeline first" not in html_str

    def test_render_no_available_conditions_shows_hint(self) -> None:
        result = render_condition_windows_editor("inst-4", None)
        html_str = str(result)
        assert "No upstream SNIRF" in html_str

    def test_switch_off_hides_table(self) -> None:
        result = render_condition_windows_editor("inst-5", {})
        html_str = str(result)
        assert "'display': 'none'" in html_str

    def test_switch_on_shows_table(self) -> None:
        current = {
            "Rest": {
                "tmin": -2.0,
                "tmax": 18.0,
                "baseline_tmin": -2.0,
                "baseline_tmax": 0.0,
            }
        }
        result = render_condition_windows_editor("inst-6", current)
        html_str = str(result)
        assert "'display': 'block'" in html_str

    def test_add_button_present(self) -> None:
        result = render_condition_windows_editor("inst-7", None)
        html_str = str(result)
        assert "cond-window-add" in html_str
        assert "Add condition" in html_str

    def test_remove_button_present_with_rows(self) -> None:
        current = {
            "Cond1": {
                "tmin": -2.0,
                "tmax": 18.0,
                "baseline_tmin": -2.0,
                "baseline_tmax": 0.0,
            }
        }
        result = render_condition_windows_editor("inst-8", current)
        html_str = str(result)
        assert "cond-window-remove" in html_str
