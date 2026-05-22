"""Tests for the guided tutorial overlay (T-021).

Covers:
  - TutorialStep dataclass fields
  - TUTORIAL_STEPS count and required attributes
  - render_tutorial_modal output for each step
  - Tutorial store initial state
  - Target IDs reference real layout IDs
  - Template pipeline auto-loading
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc

from nirspy.gui.components.tutorial import (
    TUTORIAL_CSS,
    TUTORIAL_HIGHLIGHT_CLASS,
    TUTORIAL_STEPS,
    TutorialStep,
    render_tutorial_modal,
)


class TestTutorialSteps:
    """Validate the TUTORIAL_STEPS list."""

    def test_steps_count(self) -> None:
        assert len(TUTORIAL_STEPS) == 5

    def test_steps_are_1_indexed(self) -> None:
        for i, ts in enumerate(TUTORIAL_STEPS):
            assert ts.step == i + 1

    def test_each_step_has_required_fields(self) -> None:
        for ts in TUTORIAL_STEPS:
            assert isinstance(ts, TutorialStep)
            assert ts.title
            assert ts.body
            assert ts.target_id

    def test_steps_are_frozen(self) -> None:
        ts = TUTORIAL_STEPS[0]
        try:
            ts.step = 99  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")  # noqa: TRY301
        except AttributeError:
            pass

    def test_target_ids_are_unique(self) -> None:
        ids = [ts.target_id for ts in TUTORIAL_STEPS]
        assert len(ids) == len(set(ids))


class TestTutorialTargetIds:
    """Verify that target_ids reference elements in the layout."""

    _KNOWN_LAYOUT_IDS = {
        "block-catalog",
        "pipeline-view",
        "param-editor",
        "run-button",
        "viz-tabs",
    }

    def test_all_target_ids_exist_in_layout(self) -> None:
        for ts in TUTORIAL_STEPS:
            assert ts.target_id in self._KNOWN_LAYOUT_IDS, (
                f"target_id '{ts.target_id}' not found in layout"
            )


class TestRenderTutorialModal:
    """Validate render_tutorial_modal output."""

    def test_renders_modal_for_each_step(self) -> None:
        for i in range(len(TUTORIAL_STEPS)):
            modal = render_tutorial_modal(i)
            assert isinstance(modal, dbc.Modal)
            assert modal.id == "tutorial-modal"

    def test_modal_is_open(self) -> None:
        modal = render_tutorial_modal(0)
        assert modal.is_open is True

    def test_modal_has_static_backdrop(self) -> None:
        modal = render_tutorial_modal(0)
        assert modal.backdrop == "static"

    def test_first_step_prev_disabled(self) -> None:
        modal = render_tutorial_modal(0)
        footer = modal.children[2]  # ModalFooter
        prev_btn = footer.children[1]  # Previous button
        assert prev_btn.disabled is True

    def test_last_step_has_finish_button(self) -> None:
        modal = render_tutorial_modal(len(TUTORIAL_STEPS) - 1)
        footer = modal.children[2]
        last_btn = footer.children[2]  # Finish button
        assert last_btn.id == "tutorial-finish"

    def test_middle_step_has_next_button(self) -> None:
        modal = render_tutorial_modal(1)
        footer = modal.children[2]
        next_btn = footer.children[2]
        assert next_btn.id == "tutorial-next"

    def test_skip_button_always_present(self) -> None:
        for i in range(len(TUTORIAL_STEPS)):
            modal = render_tutorial_modal(i)
            footer = modal.children[2]
            skip_btn = footer.children[0]
            assert skip_btn.id == "tutorial-skip"

    def test_invalid_index_defaults_to_first(self) -> None:
        modal = render_tutorial_modal(-1)
        header = modal.children[0]
        assert "Step 1/" in str(header.children)

    def test_step_counter_in_header(self) -> None:
        for i, ts in enumerate(TUTORIAL_STEPS):
            modal = render_tutorial_modal(i)
            header = modal.children[0]
            expected = f"Step {ts.step}/{len(TUTORIAL_STEPS)}"
            assert expected in str(header.children)


class TestTutorialStore:
    """Validate the tutorial store initial state in layout."""

    def test_initial_store_state(self) -> None:
        from nirspy.gui.layouts import create_layout

        layout = create_layout()
        # Find the tutorial-store in the layout tree
        store = _find_component_by_id(layout, "tutorial-store")
        assert store is not None
        assert store.data == {"active": False, "step": 0}

    def test_highlight_target_store_exists(self) -> None:
        from nirspy.gui.layouts import create_layout

        layout = create_layout()
        store = _find_component_by_id(layout, "tutorial-highlight-target")
        assert store is not None
        assert store.data == ""

    def test_tutorial_button_exists(self) -> None:
        from nirspy.gui.layouts import create_layout

        layout = create_layout()
        btn = _find_component_by_id(layout, "btn-start-tutorial")
        assert btn is not None

    def test_tutorial_modal_container_exists(self) -> None:
        from nirspy.gui.layouts import create_layout

        layout = create_layout()
        container = _find_component_by_id(
            layout, "tutorial-modal-container"
        )
        assert container is not None


class TestTutorialCSS:
    """Validate the tutorial CSS string."""

    def test_highlight_class_in_css(self) -> None:
        assert TUTORIAL_HIGHLIGHT_CLASS in TUTORIAL_CSS

    def test_box_shadow_in_css(self) -> None:
        assert "box-shadow" in TUTORIAL_CSS


class TestTemplateLoading:
    """Validate the template pipeline auto-load."""

    def test_load_template_returns_list(self) -> None:
        from nirspy.gui.callbacks.tutorial_callbacks import (
            _load_template_pipeline,
        )

        result = _load_template_pipeline()
        assert isinstance(result, list)

    def test_load_template_non_empty(self) -> None:
        from nirspy.gui.callbacks.tutorial_callbacks import (
            _load_template_pipeline,
        )

        result = _load_template_pipeline()
        assert len(result) > 0, "Template should load at least one block"

    def test_load_template_entries_have_required_keys(self) -> None:
        from nirspy.gui.callbacks.tutorial_callbacks import (
            _load_template_pipeline,
        )

        result = _load_template_pipeline()
        for entry in result:
            assert "block_id" in entry
            assert "instance_id" in entry
            assert "params" in entry
            assert "enabled" in entry

    def test_load_template_first_block_is_load_snirf(self) -> None:
        from nirspy.gui.callbacks.tutorial_callbacks import (
            _load_template_pipeline,
        )

        result = _load_template_pipeline()
        assert result[0]["block_id"] == "load_snirf"


def _find_component_by_id(
    component: Any, target_id: str
) -> Any:
    """Recursively search a Dash component tree for a given id."""
    if hasattr(component, "id") and component.id == target_id:
        return component
    children = getattr(component, "children", None)
    if children is None:
        return None
    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find_component_by_id(child, target_id)
            if found is not None:
                return found
    else:
        return _find_component_by_id(children, target_id)
    return None
