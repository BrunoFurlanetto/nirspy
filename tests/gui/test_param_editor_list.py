"""Tests for list[str] multiselect in param_editor (T-011 scope D)."""

from __future__ import annotations

import dataclasses

from nirspy.blocks.manual_exclude import ManualChannelExcludeParams
from nirspy.gui.components.param_editor import _field_to_input, render_param_editor


class TestListStrMultiselect:
    """list[str] field renders multiselect when channels available."""

    def test_dropdown_when_channels_available(self) -> None:
        f = dataclasses.fields(ManualChannelExcludeParams)[0]
        components = _field_to_input(
            f, [], "inst-1", "manual_channel_exclude",
            available_channels=["S1_D1 760", "S1_D1 850", "S2_D1 760"],
        )
        html_str = str(components[0])
        assert "Dropdown" in html_str
        assert "S1_D1 760" in html_str

    def test_text_fallback_without_channels(self) -> None:
        f = dataclasses.fields(ManualChannelExcludeParams)[0]
        components = _field_to_input(
            f, [], "inst-1", "manual_channel_exclude",
        )
        html_str = str(components[0])
        assert "Run pipeline first" in html_str

    def test_existing_values_in_dropdown(self) -> None:
        f = dataclasses.fields(ManualChannelExcludeParams)[0]
        components = _field_to_input(
            f, ["S1_D1 760"], "inst-1", "manual_channel_exclude",
            available_channels=["S1_D1 760", "S1_D1 850"],
        )
        html_str = str(components[0])
        assert "S1_D1 760" in html_str

    def test_tooltip_for_channels_field(self) -> None:
        f = dataclasses.fields(ManualChannelExcludeParams)[0]
        components = _field_to_input(
            f, [], "inst-1", "manual_channel_exclude",
        )
        assert len(components) >= 2  # row + tooltip


class TestRenderEditorWithChannels:
    """render_param_editor passes available_channels through."""

    def test_with_channels(self) -> None:
        result = render_param_editor(
            "manual_channel_exclude",
            "inst-1",
            ManualChannelExcludeParams,
            {"channels": []},
            available_channels=["S1_D1 760"],
        )
        assert "Dropdown" in str(result)

    def test_without_channels(self) -> None:
        result = render_param_editor(
            "manual_channel_exclude",
            "inst-1",
            ManualChannelExcludeParams,
            {"channels": []},
        )
        assert "Run pipeline first" in str(result)
