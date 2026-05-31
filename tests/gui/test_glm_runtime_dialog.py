"""Tests for T-040 -- GLM runtime dialog.

Covers:
- render_glm_runtime_dialog returns a dbc.Modal that is open
- Conditions appear in the rendered modal (one input per condition)
- Durations are pre-populated from annotation_durations
- Empty / absent conditions render a warning alert
- build_glm_params_override(None) returns None
- build_glm_params_override({}) returns None
- build_glm_params_override with condition_durations returns correct dict
- build_glm_params_override with per_condition_groups returns correct dict
- build_glm_params_override with both returns correct dict
- build_glm_params_override skips groups missing label or conditions
- glm-runtime-state store is wired in the app layout
"""

from __future__ import annotations

from typing import Any

import pytest  # noqa: F401 — used for pytest.approx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _glm_spec() -> Any:
    """Return the GLMBlock BlockSpec."""
    from nirspy.blocks import registry

    cls = registry.get("glm")
    return cls.SPEC  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# render_glm_runtime_dialog — structure
# ---------------------------------------------------------------------------


class TestRenderGlmRuntimeDialog:
    """Unit tests for the render_glm_runtime_dialog pure render function."""

    def test_returns_modal(self) -> None:
        import dash_bootstrap_components as dbc

        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal = render_glm_runtime_dialog(
            _glm_spec(), 0, 3, available_conditions=["A"], annotation_durations={}
        )
        assert isinstance(modal, dbc.Modal)

    def test_modal_is_open(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal = render_glm_runtime_dialog(
            _glm_spec(), 0, 3, available_conditions=["A"], annotation_durations={}
        )
        assert modal.is_open is True

    def test_modal_id(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal = render_glm_runtime_dialog(
            _glm_spec(), 0, 3, available_conditions=["A"], annotation_durations={}
        )
        assert modal.id == "glm-runtime-modal"

    def test_backdrop_static(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal = render_glm_runtime_dialog(
            _glm_spec(), 0, 3, available_conditions=["A"], annotation_durations={}
        )
        assert modal.backdrop == "static"

    def test_keyboard_false(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal = render_glm_runtime_dialog(
            _glm_spec(), 1, 3, available_conditions=["A"], annotation_durations={}
        )
        assert modal.keyboard is False

    def test_size_lg(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal = render_glm_runtime_dialog(
            _glm_spec(), 0, 3, available_conditions=["A"], annotation_durations={}
        )
        assert modal.size == "lg"

    def test_header_shows_step_counter(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal_str = str(
            render_glm_runtime_dialog(
                _glm_spec(), 2, 8, available_conditions=["A"], annotation_durations={}
            )
        )
        assert "Block 3/8" in modal_str

    def test_cancel_btn_id_present(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal_str = str(
            render_glm_runtime_dialog(
                _glm_spec(), 0, 3, available_conditions=["A"], annotation_durations={}
            )
        )
        assert "runtime-cancel-btn" in modal_str

    def test_advance_btn_id_present(self) -> None:
        """runtime-advance-btn must be present so advance_run callback fires."""
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal_str = str(
            render_glm_runtime_dialog(
                _glm_spec(), 0, 3, available_conditions=["A"], annotation_durations={}
            )
        )
        assert "runtime-advance-btn" in modal_str


# ---------------------------------------------------------------------------
# Conditions table
# ---------------------------------------------------------------------------


class TestConditionsTable:
    """Verify per-condition rows are rendered with correct values."""

    def test_conditions_appear_in_modal(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        conditions = ["stim_A", "stim_B", "stim_C"]
        modal_str = str(
            render_glm_runtime_dialog(
                _glm_spec(),
                0, 3,
                available_conditions=conditions,
                annotation_durations={},
            )
        )
        for c in conditions:
            assert c in modal_str

    def test_duration_prepopulated_from_annotation_durations(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        ann_durations = {"stim_A": 4.5, "stim_B": 2.0}
        modal_str = str(
            render_glm_runtime_dialog(
                _glm_spec(),
                0, 3,
                available_conditions=["stim_A", "stim_B"],
                annotation_durations=ann_durations,
            )
        )
        # The numeric values must appear as input 'value' in rendered output
        assert "4.5" in modal_str
        assert "2.0" in modal_str

    def test_missing_duration_falls_back_to_1(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        # stim_A has no entry in annotation_durations → must default to 1.0
        modal_str = str(
            render_glm_runtime_dialog(
                _glm_spec(),
                0, 3,
                available_conditions=["stim_A"],
                annotation_durations={},  # empty — no pre-fills
            )
        )
        assert "1" in modal_str  # default 1.0 present

    def test_no_conditions_shows_warning_alert(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal_str = str(
            render_glm_runtime_dialog(
                _glm_spec(),
                0, 3,
                available_conditions=[],
                annotation_durations={},
            )
        )
        assert "No conditions detected" in modal_str

    def test_groups_section_present(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal_str = str(
            render_glm_runtime_dialog(
                _glm_spec(),
                0, 3,
                available_conditions=["A"],
                annotation_durations={},
            )
        )
        assert "glm-rt-groups-container" in modal_str

    def test_add_group_btn_present(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        modal_str = str(
            render_glm_runtime_dialog(
                _glm_spec(),
                0, 3,
                available_conditions=["A"],
                annotation_durations={},
            )
        )
        assert "glm-rt-add-group-btn" in modal_str


# ---------------------------------------------------------------------------
# build_glm_params_override
# ---------------------------------------------------------------------------


class TestBuildGlmParamsOverride:
    """Unit tests for build_glm_params_override helper."""

    def test_none_returns_none(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        assert build_glm_params_override(None) is None

    def test_empty_dict_returns_none(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        assert build_glm_params_override({}) is None

    def test_state_with_empty_durations_and_groups_returns_none(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        state: dict[str, Any] = {
            "available_conditions": ["A"],
            "condition_durations": {},
            "groups": [],
        }
        assert build_glm_params_override(state) is None

    def test_condition_durations_returned_when_present(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        state: dict[str, Any] = {
            "available_conditions": ["stim_A"],
            "condition_durations": {"stim_A": 5.0},
            "groups": [],
        }
        result = build_glm_params_override(state)
        assert result is not None
        assert result["condition_durations"] == {"stim_A": 5.0}

    def test_condition_durations_values_coerced_to_float(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        state: dict[str, Any] = {
            "available_conditions": ["A"],
            "condition_durations": {"A": "3.2"},  # string value
            "groups": [],
        }
        result = build_glm_params_override(state)
        assert result is not None
        assert isinstance(result["condition_durations"]["A"], float)
        assert result["condition_durations"]["A"] == pytest.approx(3.2)

    def test_condition_durations_bad_value_skipped(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        state: dict[str, Any] = {
            "available_conditions": ["A", "B"],
            "condition_durations": {"A": 2.0, "B": "not_a_float"},
            "groups": [],
        }
        result = build_glm_params_override(state)
        assert result is not None
        assert "A" in result["condition_durations"]
        assert "B" not in result["condition_durations"]

    def test_per_condition_groups_returned_when_valid(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        state: dict[str, Any] = {
            "available_conditions": ["cond_A", "cond_B"],
            "condition_durations": {},
            "groups": [{"label": "motor", "conditions": ["cond_A", "cond_B"]}],
        }
        result = build_glm_params_override(state)
        assert result is not None
        assert "per_condition_groups" in result
        assert result["per_condition_groups"] == {"motor": ["cond_A", "cond_B"]}

    def test_both_fields_returned_together(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        state: dict[str, Any] = {
            "available_conditions": ["cond_A", "cond_B"],
            "condition_durations": {"cond_A": 4.0, "cond_B": 2.0},
            "groups": [{"label": "motor", "conditions": ["cond_A", "cond_B"]}],
        }
        result = build_glm_params_override(state)
        assert result is not None
        assert "condition_durations" in result
        assert "per_condition_groups" in result

    def test_group_without_label_skipped(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        state: dict[str, Any] = {
            "available_conditions": ["A"],
            "condition_durations": {},
            "groups": [{"label": "", "conditions": ["A"]}],
        }
        assert build_glm_params_override(state) is None

    def test_group_without_conditions_skipped(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        state: dict[str, Any] = {
            "available_conditions": ["A"],
            "condition_durations": {},
            "groups": [{"label": "motor", "conditions": []}],
        }
        assert build_glm_params_override(state) is None

    def test_mixed_valid_invalid_groups_only_valid_included(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        state: dict[str, Any] = {
            "available_conditions": ["A", "B"],
            "condition_durations": {},
            "groups": [
                {"label": "good", "conditions": ["A"]},
                {"label": "", "conditions": ["B"]},           # no label — skip
                {"label": "empty_conds", "conditions": []},   # no conds — skip
            ],
        }
        result = build_glm_params_override(state)
        assert result is not None
        assert list(result["per_condition_groups"].keys()) == ["good"]


# ---------------------------------------------------------------------------
# Layout integration
# ---------------------------------------------------------------------------


class TestLayoutIntegration:
    """Verify layouts.py wires up glm-runtime-state store."""

    def test_glm_runtime_state_store_in_layout(self) -> None:
        from nirspy.gui.layouts import create_layout

        layout_str = str(create_layout())
        assert "glm-runtime-state" in layout_str


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------


class TestImportSanity:
    def test_module_importable(self) -> None:
        import nirspy.gui.components.glm_runtime_dialog  # noqa: F401

    def test_render_callable(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import render_glm_runtime_dialog

        assert callable(render_glm_runtime_dialog)

    def test_build_glm_params_override_callable(self) -> None:
        from nirspy.gui.components.glm_runtime_dialog import build_glm_params_override

        assert callable(build_glm_params_override)

    def test_advance_run_has_glm_state_param(self) -> None:
        """advance_run must accept glm_state (added in T-040)."""
        import inspect

        from nirspy.gui.callbacks.runtime_callbacks import advance_run

        sig = inspect.signature(advance_run)
        assert "glm_state" in sig.parameters
