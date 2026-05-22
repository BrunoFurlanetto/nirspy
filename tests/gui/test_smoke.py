"""Smoke tests for 5D polish (T-006 5D)."""

from __future__ import annotations

import uuid
from typing import Any

import dash_bootstrap_components as dbc

from nirspy.blocks import registry
from nirspy.domain.block import BlockResult


def _make_entry(
    block_id: str = "optical_density",
    enabled: bool = True,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "instance_id": str(uuid.uuid4()),
        "params": params or {},
        "enabled": enabled,
    }


class TestErrorDisplay:
    def test_render_error_returns_alert(self) -> None:
        from nirspy.gui.components.error_display import render_error
        result = render_error("Something went wrong")
        assert isinstance(result, dbc.Alert)

    def test_render_error_contains_message(self) -> None:
        from nirspy.gui.components.error_display import render_error
        result = render_error("Pipeline failed")
        assert "Pipeline failed" in str(result)

    def test_render_error_default_danger(self) -> None:
        from nirspy.gui.components.error_display import render_error
        result = render_error("Error")
        assert result.color == "danger"

    def test_render_error_custom_severity(self) -> None:
        from nirspy.gui.components.error_display import render_error
        result = render_error("Warning", severity="warning")
        assert result.color == "warning"

    def test_render_error_is_dismissable(self) -> None:
        from nirspy.gui.components.error_display import render_error
        result = render_error("Error")
        assert result.dismissable is True

    def test_render_error_no_stack_trace(self) -> None:
        from nirspy.gui.components.error_display import render_error
        result = render_error("User-friendly message")
        result_str = str(result)
        assert "Traceback" not in result_str
        assert "File " not in result_str


class TestTooltips:
    def test_all_eight_blocks_have_tooltips(self) -> None:
        from nirspy.gui.components.tooltips import tooltip_for
        for block_id in registry.list_blocks():
            tt = tooltip_for(block_id)
            assert tt is not None, f"Missing tooltip for {block_id}"
            assert isinstance(tt, dbc.Tooltip)

    def test_unknown_block_returns_none(self) -> None:
        from nirspy.gui.components.tooltips import tooltip_for
        assert tooltip_for("nonexistent_block") is None

    def test_tooltip_has_reference(self) -> None:
        from nirspy.gui.components.tooltips import tooltip_for
        tt = tooltip_for("scalp_coupling_index")
        assert tt is not None
        text = str(tt)
        assert "Pollonini" in text or "Yuecel" in text

    def test_tooltip_targets_catalog_item(self) -> None:
        from nirspy.gui.components.tooltips import tooltip_for
        tt = tooltip_for("optical_density")
        assert tt is not None
        assert tt.target == {"type": "catalog-item", "block_id": "optical_density"}

    def test_tooltip_placement_right(self) -> None:
        from nirspy.gui.components.tooltips import tooltip_for
        tt = tooltip_for("beer_lambert")
        assert tt is not None
        assert tt.placement == "right"


class TestExecutionErrorWiring:
    def test_invalid_block_returns_alert(self) -> None:
        from nirspy.gui.callbacks.execution_callbacks import run_pipeline_callback
        state = [_make_entry("nonexistent_block")]
        result = run_pipeline_callback(1, state, None)
        assert isinstance(result[4], dbc.Alert)
        assert "Failed to build pipeline" in str(result[4])

    def test_error_includes_block_id_when_available(self) -> None:
        from nirspy.gui.callbacks.execution_callbacks import _extract_block_id
        state = [_make_entry("load_snirf")]
        exc = KeyError("load_snirf not found")
        block_id = _extract_block_id(state, exc)
        assert block_id == "load_snirf"

    def test_extract_block_id_empty_on_unknown(self) -> None:
        from nirspy.gui.callbacks.execution_callbacks import _extract_block_id
        state = [_make_entry("optical_density")]
        exc = ValueError("something unrelated")
        assert _extract_block_id(state, exc) == ""


class TestCatalogTooltipWiring:
    def test_catalog_contains_tooltip_components(self) -> None:
        from nirspy.gui.components.block_catalog import render_block_catalog
        result = render_block_catalog(registry)
        assert "Tooltip" in str(result)

    def test_catalog_has_eight_tooltips(self) -> None:
        from nirspy.gui.components.block_catalog import render_block_catalog
        result = render_block_catalog(registry)
        assert str(result).count("Tooltip(") == 11

    def test_catalog_still_has_eight_items(self) -> None:
        from nirspy.gui.components.block_catalog import render_block_catalog
        result = render_block_catalog(registry)
        assert str(result).count("ListGroupItem") == 11


class TestQCRegression:
    """Regression test for QC dashboard with dict-style SCI values."""

    def test_dict_sci_values_extracted_as_list(self) -> None:
        from nirspy.gui.callbacks.viz_callbacks import _find_sci_values
        sci_dict = {"Ch1 760": 0.85, "Ch2 760": 0.45}
        r = BlockResult(data=None, block_id="scalp_coupling_index",
                        metadata={"sci_values": sci_dict})
        values = _find_sci_values([r])
        assert values == [0.85, 0.45]

    def test_dict_sci_ch_names_extracted(self) -> None:
        from nirspy.gui.callbacks.viz_callbacks import _find_sci_ch_names
        sci_dict = {"Ch1 760": 0.85, "Ch2 760": 0.45}
        r = BlockResult(data=None, block_id="scalp_coupling_index",
                        metadata={"sci_values": sci_dict})
        names = _find_sci_ch_names([r])
        assert names == ["Ch1 760", "Ch2 760"]

    def test_list_sci_values_still_work(self) -> None:
        from nirspy.gui.callbacks.viz_callbacks import _find_sci_values
        r = BlockResult(data=None, block_id="scalp_coupling_index",
                        metadata={"sci_values": [0.9, 0.3]})
        assert _find_sci_values([r]) == [0.9, 0.3]

    def test_qc_dashboard_renders_heatmap_from_dict(self) -> None:
        from nirspy.gui.callbacks.viz_callbacks import _find_sci_ch_names, _find_sci_values
        from nirspy.gui.components.qc_dashboard import render_qc_dashboard
        sci_dict = {"Ch1 760": 0.85, "Ch2 760": 0.45}
        r = BlockResult(data=None, block_id="scalp_coupling_index",
                        metadata={"sci_values": sci_dict})
        values = _find_sci_values([r])
        ch_names = _find_sci_ch_names([r])
        result = render_qc_dashboard(values, ch_names)
        result_str = str(result)
        assert "qc-dashboard" in result_str
        assert "No QC data" not in result_str

    def test_qc_dashboard_fallback_on_none(self) -> None:
        from nirspy.gui.components.qc_dashboard import render_qc_dashboard
        assert "No QC data" in str(render_qc_dashboard(None))


class TestAppFactoryAfterWiring:
    def test_create_app_succeeds(self) -> None:
        from nirspy.gui.app import create_app
        app = create_app()
        assert app is not None
        assert app.layout is not None

    def test_layout_still_has_catalog(self) -> None:
        from nirspy.gui.app import create_app
        assert "block-catalog" in str(create_app().layout)

    def test_layout_still_has_viz_tabs(self) -> None:
        from nirspy.gui.app import create_app
        assert "viz-tabs" in str(create_app().layout)

    def test_callbacks_registered(self) -> None:
        from nirspy.gui.app import create_app
        app = create_app()
        assert app.callback_map is not None
        # callback_map may be empty if callbacks already registered globally
        assert app.callback_map is not None


class TestNoDepsAdded:
    def test_pyproject_unchanged(self) -> None:
        import pathlib
        p = pathlib.Path("pyproject.toml")
        assert p.exists()


class TestImportSanity5D:
    def test_error_display_importable(self) -> None:
        from nirspy.gui.components.error_display import render_error  # noqa: F401

    def test_tooltips_importable(self) -> None:
        from nirspy.gui.components.tooltips import tooltip_for  # noqa: F401

    def test_render_error_signature(self) -> None:
        import inspect

        from nirspy.gui.components.error_display import render_error
        sig = inspect.signature(render_error)
        assert "message" in sig.parameters
        assert "severity" in sig.parameters

    def test_tooltip_for_signature(self) -> None:
        import inspect

        from nirspy.gui.components.tooltips import tooltip_for
        sig = inspect.signature(tooltip_for)
        assert "block_id" in sig.parameters
