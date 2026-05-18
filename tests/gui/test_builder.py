"""Tests for the pipeline builder GUI (T-006 5B)."""

from __future__ import annotations

import base64
import dataclasses
import uuid
from typing import Any

import yaml

from nirspy.blocks import registry
from nirspy.domain.block import BlockSpec
from nirspy.gui.components.block_card import render_block_card
from nirspy.gui.components.block_catalog import render_block_catalog
from nirspy.gui.components.param_editor import render_param_editor
from nirspy.gui.components.pipeline_view import render_pipeline_view


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


class TestBlockCatalog:
    def test_renders_all_registered_blocks(self) -> None:
        result = render_block_catalog(registry)
        html_str = str(result)
        for block_id in registry.list_blocks():
            block_cls = registry.get(block_id)
            spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]
            assert spec.display_name in html_str

    def test_catalog_has_eight_blocks(self) -> None:
        result = render_block_catalog(registry)
        html_str = str(result)
        # Tooltips also target catalog-item IDs, so count ListGroupItem occurrences
        assert html_str.count("ListGroupItem") == 8

    def test_catalog_shows_io_types(self) -> None:
        result = render_block_catalog(registry)
        assert "raw" in str(result).lower()


class TestBlockCard:
    def test_card_shows_display_name(self) -> None:
        result = render_block_card(
            block_id="optical_density", instance_id="test-id",
            display_name="Optical Density", input_type="raw",
            output_type="raw_od", enabled=True, selected=False,
            is_first=True, is_last=True,
        )
        assert "Optical Density" in str(result)

    def test_card_dimmed_when_disabled(self) -> None:
        result = render_block_card(
            block_id="optical_density", instance_id="test-id",
            display_name="Optical Density", input_type="raw",
            output_type="raw_od", enabled=False, selected=False,
            is_first=True, is_last=True,
        )
        assert "0.45" in str(result)

    def test_card_selected_highlight(self) -> None:
        result = render_block_card(
            block_id="optical_density", instance_id="test-id",
            display_name="Optical Density", input_type="raw",
            output_type="raw_od", enabled=True, selected=True,
            is_first=True, is_last=True,
        )
        assert "primary" in str(result)

    def test_compat_green_when_compatible(self) -> None:
        result = render_block_card(
            block_id="beer_lambert", instance_id="test-id",
            display_name="Beer Lambert", input_type="raw_od",
            output_type="raw_haemo", enabled=True, selected=False,
            is_first=False, is_last=True, prev_output_type="raw_od",
        )
        assert "#28a745" in str(result)

    def test_compat_red_when_incompatible(self) -> None:
        result = render_block_card(
            block_id="beer_lambert", instance_id="test-id",
            display_name="Beer Lambert", input_type="raw_od",
            output_type="raw_haemo", enabled=True, selected=False,
            is_first=False, is_last=True, prev_output_type="raw_haemo",
        )
        assert "#dc3545" in str(result)

    def test_compat_green_when_any(self) -> None:
        result = render_block_card(
            block_id="bandpass_filter", instance_id="test-id",
            display_name="Bandpass Filter", input_type="any",
            output_type="any", enabled=True, selected=False,
            is_first=False, is_last=True, prev_output_type="raw_haemo",
        )
        assert "#28a745" in str(result)


class TestPipelineView:
    def test_empty_state_shows_placeholder(self) -> None:
        result = render_pipeline_view([], None)
        assert "Add blocks" in str(result)

    def test_renders_blocks_from_state(self) -> None:
        state = [_make_entry("optical_density"), _make_entry("beer_lambert")]
        result = render_pipeline_view(state, None)
        html_str = str(result)
        assert "Optical Density" in html_str
        assert "Modified Beer-Lambert" in html_str

    def test_selected_block_highlighted(self) -> None:
        entry = _make_entry("optical_density")
        result = render_pipeline_view([entry], entry["instance_id"])
        assert "primary" in str(result)


class TestParamEditor:
    def test_no_block_selected(self) -> None:
        result = render_param_editor(
            block_id=None, instance_id=None,
            params_class=None, current_values={},
        )
        assert "Select a block" in str(result)

    def test_no_params_shows_message(self) -> None:
        from nirspy.blocks.preprocessing import OpticalDensityParams
        result = render_param_editor(
            block_id="optical_density", instance_id="test-id",
            params_class=OpticalDensityParams, current_values={},
        )
        assert "No parameters" in str(result)

    def test_float_param_number_input(self) -> None:
        from nirspy.blocks.preprocessing import BeerLambertParams
        result = render_param_editor(
            block_id="beer_lambert", instance_id="test-id",
            params_class=BeerLambertParams, current_values={"ppf": 6.0},
        )
        html_str = str(result)
        assert "ppf" in html_str
        assert "number" in html_str

    def test_str_param_text_input(self) -> None:
        from nirspy.blocks.load import LoadSnirfParams
        result = render_param_editor(
            block_id="load_snirf", instance_id="test-id",
            params_class=LoadSnirfParams,
            current_values={"path": "/data/test.snirf"},
        )
        html_str = str(result)
        assert "path" in html_str
        assert "text" in html_str

    def test_bool_param_checkbox(self) -> None:
        from nirspy.blocks.analysis import BlockAverageParams
        result = render_param_editor(
            block_id="block_average", instance_id="test-id",
            params_class=BlockAverageParams,
            current_values={"reject_by_amplitude": True},
        )
        assert "reject_by_amplitude" in str(result)

    def test_uses_dataclass_fields(self) -> None:
        from nirspy.blocks.preprocessing import BeerLambertParams
        fields = dataclasses.fields(BeerLambertParams)
        result = render_param_editor(
            block_id="beer_lambert", instance_id="test-id",
            params_class=BeerLambertParams, current_values={},
        )
        html_str = str(result)
        for f in fields:
            assert f.name in html_str

    def test_block_average_all_fields(self) -> None:
        from nirspy.blocks.analysis import BlockAverageParams
        fields = dataclasses.fields(BlockAverageParams)
        result = render_param_editor(
            block_id="block_average", instance_id="test-id",
            params_class=BlockAverageParams, current_values={},
        )
        html_str = str(result)
        for f in fields:
            if f.name == "per_condition_windows":
                # Custom widget; check for its switch ID instead
                assert "cond-window-switch" in html_str
            else:
                assert f.name in html_str


class TestIOSerialization:
    def test_state_to_yaml_valid_schema(self) -> None:
        state = [
            _make_entry("optical_density"),
            _make_entry("beer_lambert", params={"ppf": 6.0}),
        ]
        params_map: dict[str, Any] = {}
        steps: list[dict[str, Any]] = []
        for entry in state:
            steps.append({"block_id": entry["block_id"], "enabled": True})
            p = entry.get("params", {})
            if p:
                params_map[entry["block_id"]] = p
        data = {
            "description": "",
            "name": "test",
            "params": params_map,
            "schema_version": "0.1",
            "steps": steps,
        }
        yaml_text = yaml.dump(data, sort_keys=True)
        parsed = yaml.safe_load(yaml_text)
        assert parsed["schema_version"] == "0.1"
        assert len(parsed["steps"]) == 2
        assert parsed["params"]["beer_lambert"]["ppf"] == 6.0

    def test_yaml_round_trip_decode(self) -> None:
        yaml_text = yaml.dump({
            "description": "",
            "name": "test",
            "params": {"beer_lambert": {"ppf": 5.0}},
            "schema_version": "0.1",
            "steps": [
                {"block_id": "optical_density", "enabled": True},
                {"block_id": "beer_lambert", "enabled": True},
            ],
        }, sort_keys=True)
        encoded = base64.b64encode(yaml_text.encode("utf-8")).decode("utf-8")
        contents = "data:application/x-yaml;base64," + encoded
        content_string = contents.split(",", 1)[1]
        decoded = base64.b64decode(content_string).decode("utf-8")
        data: dict[str, Any] = yaml.safe_load(decoded)
        assert data["schema_version"] == "0.1"
        assert len(data["steps"]) == 2

    def test_load_fills_defaults(self) -> None:
        yaml_text = yaml.dump({
            "description": "",
            "name": "test",
            "params": {},
            "schema_version": "0.1",
            "steps": [{"block_id": "beer_lambert", "enabled": True}],
        })
        data = yaml.safe_load(yaml_text)
        for step in data.get("steps", []):
            block_cls = registry.get(step["block_id"])
            spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]
            if spec.params_class and dataclasses.is_dataclass(spec.params_class):
                try:
                    obj = spec.params_class()
                    params = dataclasses.asdict(obj)
                    assert params["ppf"] == 6.0
                except TypeError:
                    pass


class TestLayoutStructure:
    def test_layout_has_download(self) -> None:
        from nirspy.gui.app import create_app
        assert "download-pipeline" in str(create_app().layout)

    def test_layout_has_upload(self) -> None:
        from nirspy.gui.app import create_app
        assert "upload-pipeline" in str(create_app().layout)

    def test_layout_has_save_button(self) -> None:
        from nirspy.gui.app import create_app
        assert "btn-save-pipeline" in str(create_app().layout)

    def test_layout_catalog_shows_blocks(self) -> None:
        from nirspy.gui.app import create_app
        layout_str = str(create_app().layout)
        assert "Load SNIRF" in layout_str
        assert "Optical Density" in layout_str

    def test_app_has_callbacks(self) -> None:
        from nirspy.gui.app import create_app
        assert create_app().callback_map is not None


class TestSourceBlockIndicator:
    """GUI indicator tests for DataType.NONE source blocks (T-009)."""

    def test_source_at_pos0_no_red_dot(self) -> None:
        """Source block at position 0 should NOT show a red indicator."""
        result = render_block_card(
            block_id="load_snirf", instance_id="test-id",
            display_name="Load SNIRF", input_type="none",
            output_type="raw", enabled=True, selected=False,
            is_first=True, is_last=False,
        )
        html_str = str(result)
        assert "#dc3545" not in html_str

    def test_source_not_at_pos0_red_dot(self) -> None:
        """Source block NOT at position 0 should show a red indicator."""
        result = render_block_card(
            block_id="load_snirf", instance_id="test-id",
            display_name="Load SNIRF", input_type="none",
            output_type="raw", enabled=True, selected=False,
            is_first=False, is_last=True, prev_output_type="raw_od",
        )
        html_str = str(result)
        assert "#dc3545" in html_str
        assert "Source block must be first" in html_str

    def test_source_at_pos0_with_next_block_compatible(self) -> None:
        """Pipeline: [LoadSnirf, OD] — no red dots anywhere."""
        state = [_make_entry("load_snirf"), _make_entry("optical_density")]
        result = render_pipeline_view(state, None)
        html_str = str(result)
        # The source at pos 0 should not produce a red dot
        # OD after LoadSnirf: LoadSnirf outputs RAW, OD expects RAW — green
        assert "Source block must be first" not in html_str

    def test_any_block_still_shows_green(self) -> None:
        """BandpassFilter (ANY) after a RAW block should still show green."""
        result = render_block_card(
            block_id="bandpass_filter", instance_id="test-id",
            display_name="Bandpass Filter", input_type="any",
            output_type="any", enabled=True, selected=False,
            is_first=False, is_last=True, prev_output_type="raw",
        )
        assert "#28a745" in str(result)

    def test_load_snirf_input_type_is_none(self) -> None:
        """LoadSnirfBlock.SPEC.input_type must be DataType.NONE."""
        from nirspy.blocks.load import LoadSnirfBlock
        from nirspy.domain.data_types import DataType
        assert LoadSnirfBlock.SPEC.input_type is DataType.NONE

    def test_load_snirf_input_type_not_any(self) -> None:
        """LoadSnirfBlock.SPEC.input_type must NOT be DataType.ANY."""
        from nirspy.blocks.load import LoadSnirfBlock
        from nirspy.domain.data_types import DataType
        assert LoadSnirfBlock.SPEC.input_type is not DataType.ANY
