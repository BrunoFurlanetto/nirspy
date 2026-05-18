"""Tests for ParamMeta integration in param_editor (T-011 scope C)."""

from __future__ import annotations

from dataclasses import dataclass

import dash_bootstrap_components as dbc

from nirspy.gui.components.param_editor import (
    _field_to_input,
    _resolve_field_type,
    render_param_editor,
)
from nirspy.gui.components.param_metadata import metadata_for


class TestParamMetaLookup:
    """metadata_for returns correct entries for registered fields."""

    def test_beer_lambert_ppf(self) -> None:
        meta = metadata_for("beer_lambert", "ppf")
        assert meta is not None
        assert "DPF" in meta.label
        assert meta.min == 3
        assert meta.max == 8

    def test_bandpass_l_freq(self) -> None:
        meta = metadata_for("bandpass_filter", "l_freq")
        assert meta is not None
        assert meta.unit == "Hz"

    def test_prune_sci_threshold(self) -> None:
        meta = metadata_for("prune_channels", "sci_threshold")
        assert meta is not None
        assert meta.reference is not None
        assert "Pollonini" in meta.reference

    def test_block_average_tmin(self) -> None:
        meta = metadata_for("block_average", "tmin")
        assert meta is not None
        assert meta.unit == "s"

    def test_unknown_field_returns_none(self) -> None:
        assert metadata_for("beer_lambert", "nonexistent") is None

    def test_unknown_block_returns_none(self) -> None:
        assert metadata_for("nonexistent_block", "ppf") is None


class TestParamMetaInEditor:
    """ParamMeta is consumed by _field_to_input for labels, tooltips, attrs."""

    def test_label_uses_meta_label(self) -> None:
        from nirspy.blocks.preprocessing import BeerLambertParams
        import dataclasses
        ppf_field = dataclasses.fields(BeerLambertParams)[0]
        components = _field_to_input(ppf_field, 6.0, "inst-1", "beer_lambert")
        html_str = str(components[0])
        assert "DPF" in html_str

    def test_tooltip_rendered_for_known_field(self) -> None:
        from nirspy.blocks.preprocessing import BeerLambertParams
        import dataclasses
        ppf_field = dataclasses.fields(BeerLambertParams)[0]
        components = _field_to_input(ppf_field, 6.0, "inst-1", "beer_lambert")
        assert len(components) >= 2
        tooltip = components[1]
        assert isinstance(tooltip, dbc.Tooltip)

    def test_no_tooltip_for_unknown_field(self) -> None:
        @dataclass(frozen=True)
        class FakeParams:
            mystery: float = 1.0
        import dataclasses
        f = dataclasses.fields(FakeParams)[0]
        components = _field_to_input(f, 1.0, "inst-1", "fake_block")
        assert len(components) == 1

    def test_min_max_on_input(self) -> None:
        from nirspy.blocks.preprocessing import BeerLambertParams
        import dataclasses
        ppf_field = dataclasses.fields(BeerLambertParams)[0]
        components = _field_to_input(ppf_field, 6.0, "inst-1", "beer_lambert")
        html_str = str(components[0])
        assert "3" in html_str
        assert "8" in html_str

    def test_unit_in_label(self) -> None:
        from nirspy.blocks.preprocessing import BandpassFilterParams
        import dataclasses
        l_freq_field = dataclasses.fields(BandpassFilterParams)[0]
        components = _field_to_input(l_freq_field, 0.01, "inst-1", "bandpass_filter")
        assert "Hz" in str(components[0])


class TestOptionalFloatCheckbox:
    """Optional[float] fields get use-default checkbox."""

    def test_optional_float_has_checkbox(self) -> None:
        from nirspy.blocks.preprocessing import BandpassFilterParams
        import dataclasses
        l_freq_field = dataclasses.fields(BandpassFilterParams)[0]
        components = _field_to_input(l_freq_field, None, "inst-1", "bandpass_filter")
        html_str = str(components[0])
        assert "use default" in html_str

    def test_optional_float_disabled_when_none(self) -> None:
        from nirspy.blocks.preprocessing import BandpassFilterParams
        import dataclasses
        l_freq_field = dataclasses.fields(BandpassFilterParams)[0]
        components = _field_to_input(l_freq_field, None, "inst-1", "bandpass_filter")
        html_str = str(components[0])
        assert "True" in html_str  # disabled=True or checkbox value=True


class TestResolveFieldTypeList:

    def test_list_str_string_annotation(self) -> None:
        is_opt, resolved = _resolve_field_type("list[str]")
        assert resolved is list
        assert not is_opt

    def test_list_str_optional_string(self) -> None:
        is_opt, resolved = _resolve_field_type("list[str] | None")
        assert resolved is list
        assert is_opt

    def test_float_still_works(self) -> None:
        is_opt, resolved = _resolve_field_type("float")
        assert resolved is float
        assert not is_opt


class TestRenderParamEditorBackcompat:

    def test_none_block_placeholder(self) -> None:
        result = render_param_editor(None, None, None, {})
        assert "Select a block" in str(result)

    def test_beer_lambert_renders_with_meta(self) -> None:
        from nirspy.blocks.preprocessing import BeerLambertParams
        result = render_param_editor(
            "beer_lambert", "inst-1", BeerLambertParams, {"ppf": 6.0}
        )
        assert "DPF" in str(result)
