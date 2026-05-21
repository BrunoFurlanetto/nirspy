"""Tests for the converter GUI -- tab, upload, convert callbacks."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nirspy.gui.components.converter_view import render_converter_tab


class TestConverterTabLayout:
    """Component IDs present in converter tab."""

    def test_tab_has_upload(self) -> None:
        tab = render_converter_tab()
        ids = _collect_ids(tab)
        assert "converter-upload" in ids

    def test_tab_has_direction(self) -> None:
        tab = render_converter_tab()
        ids = _collect_ids(tab)
        assert "converter-direction" in ids

    def test_tab_has_strip_pii(self) -> None:
        tab = render_converter_tab()
        ids = _collect_ids(tab)
        assert "converter-strip-pii" in ids

    def test_tab_has_convert_button(self) -> None:
        tab = render_converter_tab()
        ids = _collect_ids(tab)
        assert "converter-btn-convert" in ids

    def test_tab_has_download(self) -> None:
        tab = render_converter_tab()
        ids = _collect_ids(tab)
        assert "converter-download" in ids

    def test_tab_has_status(self) -> None:
        tab = render_converter_tab()
        ids = _collect_ids(tab)
        assert "converter-status" in ids

    def test_tab_has_filename(self) -> None:
        tab = render_converter_tab()
        ids = _collect_ids(tab)
        assert "converter-filename" in ids

    def test_button_disabled_default(self) -> None:
        tab = render_converter_tab()
        btn = _find_component(tab, "converter-btn-convert")
        assert btn is not None
        assert btn.disabled is True


class TestConverterInLayout:
    """Convert tab in main layout."""

    def test_layout_has_convert_tab(self) -> None:
        from nirspy.gui.layouts import create_layout

        layout = create_layout()
        assert "tab-convert" in str(layout)

    def test_layout_has_converter_upload(self) -> None:
        from nirspy.gui.layouts import create_layout

        layout = create_layout()
        ids = _collect_ids(layout)
        assert "converter-upload" in ids


class TestOnFileUpload:
    """Upload callback displays filename and enables button."""

    def test_no_filename_disabled(self) -> None:
        from nirspy.gui.callbacks.converter_callbacks import on_file_upload

        children, disabled = on_file_upload(None)
        assert children == []
        assert disabled is True

    def test_nirs_enables(self) -> None:
        from nirspy.gui.callbacks.converter_callbacks import on_file_upload

        _ch, disabled = on_file_upload("sample.nirs")
        assert disabled is False

    def test_snirf_enables(self) -> None:
        from nirspy.gui.callbacks.converter_callbacks import on_file_upload

        _ch, disabled = on_file_upload("data.snirf")
        assert disabled is False

    def test_txt_primary_badge(self) -> None:
        from nirspy.gui.callbacks.converter_callbacks import on_file_upload

        children, disabled = on_file_upload("data.txt")
        assert disabled is False
        assert children.color == "primary"


class TestOnConvert:
    """Conversion callback logic."""

    def test_no_clicks_prevent_update(self) -> None:
        from dash.exceptions import PreventUpdate

        from nirspy.gui.callbacks.converter_callbacks import on_convert

        with pytest.raises(PreventUpdate):
            on_convert(None, "data", "f.nirs", "nirs_to_snirf", False)

    def test_no_contents_prevent_update(self) -> None:
        from dash.exceptions import PreventUpdate

        from nirspy.gui.callbacks.converter_callbacks import on_convert

        with pytest.raises(PreventUpdate):
            on_convert(1, None, "f.nirs", "nirs_to_snirf", False)

    def test_extension_mismatch_error(self) -> None:
        from nirspy.gui.callbacks.converter_callbacks import on_convert

        fake = "data:;base64," + base64.b64encode(b"fake").decode()
        status, _dl = on_convert(1, fake, "data.snirf", "nirs_to_snirf", False)
        assert status.color == "danger"

    def test_invalid_direction_error(self) -> None:
        from nirspy.gui.callbacks.converter_callbacks import on_convert

        fake = "data:;base64," + base64.b64encode(b"fake").decode()
        status, _dl = on_convert(1, fake, "data.nirs", "invalid_dir", False)
        assert status.color == "danger"

    @patch("nirspy.io.converters.nirs_to_snirf")
    def test_success_triggers_download(self, mock_conv: MagicMock) -> None:
        from nirspy.gui.callbacks.converter_callbacks import on_convert

        def fake_conv(inp: Any, out: Any, **kw: Any) -> None:
            Path(out).write_bytes(b"converted")

        mock_conv.side_effect = fake_conv
        fake = "data:;base64," + base64.b64encode(b"nirs").decode()
        status, download = on_convert(1, fake, "sample.nirs", "nirs_to_snirf", False)
        assert status.color == "success"
        assert download is not None
        assert "sample.snirf" in str(download)
        mock_conv.assert_called_once()

    @patch("nirspy.io.converters.nirs_to_snirf")
    def test_converter_error_friendly(self, mock_conv: MagicMock) -> None:
        from nirspy.domain.exceptions import ConverterError
        from nirspy.gui.callbacks.converter_callbacks import on_convert

        mock_conv.side_effect = ConverterError("File is corrupted")
        fake = "data:;base64," + base64.b64encode(b"bad").decode()
        status, _dl = on_convert(1, fake, "bad.nirs", "nirs_to_snirf", False)
        assert status.color == "danger"
        assert "conversion failed" in status.children.lower()

    @patch("nirspy.io.converters.nirs_to_snirf")
    def test_strip_pii_forwarded(self, mock_conv: MagicMock) -> None:
        from nirspy.gui.callbacks.converter_callbacks import on_convert

        def fake_conv(inp: Any, out: Any, **kw: Any) -> None:
            Path(out).write_bytes(b"data")

        mock_conv.side_effect = fake_conv
        fake = "data:;base64," + base64.b64encode(b"data").decode()
        on_convert(1, fake, "s.nirs", "nirs_to_snirf", True)
        assert mock_conv.call_args[1].get("strip_pii") is True

    @patch("nirspy.io.converters.nirs_to_snirf")
    def test_unexpected_error_generic(self, mock_conv: MagicMock) -> None:
        from nirspy.gui.callbacks.converter_callbacks import on_convert

        mock_conv.side_effect = RuntimeError("boom")
        fake = "data:;base64," + base64.b64encode(b"data").decode()
        status, _dl = on_convert(1, fake, "f.nirs", "nirs_to_snirf", False)
        assert status.color == "danger"
        assert "unexpected" in status.children.lower()


class TestAppRegistersConverter:
    """Converter callbacks registered in the app."""

    def test_converter_callbacks_importable(self) -> None:
        from nirspy.gui.callbacks import converter_callbacks

        assert hasattr(converter_callbacks, "on_file_upload")
        assert hasattr(converter_callbacks, "on_convert")


# -- Helpers --


def _collect_ids(component: Any) -> set[str]:
    """Recursively collect all component IDs."""
    ids: set[str] = set()
    if hasattr(component, "id") and isinstance(component.id, str):
        ids.add(component.id)
    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, list):
            for child in children:
                ids.update(_collect_ids(child))
        elif children is not None:
            ids.update(_collect_ids(children))
    for attr in ("child", "data", "content"):
        val = getattr(component, attr, None)
        if val is not None and hasattr(val, "id"):
            ids.update(_collect_ids(val))
    return ids


def _find_component(component: Any, target_id: str) -> Any:
    """Find a component by ID recursively."""
    if hasattr(component, "id") and component.id == target_id:
        return component
    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, list):
            for child in children:
                result = _find_component(child, target_id)
                if result is not None:
                    return result
        elif children is not None:
            result = _find_component(children, target_id)
            if result is not None:
                return result
    return None
