"""Converter callbacks -- upload, convert, and download .nirs/.snirf files.

Handles the conversion workflow:
1. Upload sets filename display and enables the convert button.
2. Convert reads the uploaded bytes, writes a temp file, calls the
   appropriate converter, and triggers a download of the result.
3. Errors from ``ConverterError`` (and subclasses) are displayed via
   ``render_error`` -- no stack traces exposed to the user.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
from dash import (
    Input,
    Output,
    State,
    callback,
    dcc,
    no_update,
)
from dash.exceptions import PreventUpdate

from nirspy.domain.exceptions import ConverterError
from nirspy.gui.components.error_display import render_error

logger = logging.getLogger(__name__)

# Map direction -> (expected input suffix, output suffix)
_DIRECTION_META: dict[str, tuple[str, str]] = {
    "nirs_to_snirf": (".nirs", ".snirf"),
    "snirf_to_nirs": (".snirf", ".nirs"),
}


def _get_converter(direction: str) -> Any:
    """Resolve the converter function for a given direction.

    Imported lazily so that ``unittest.mock.patch`` on the module-level
    names in ``nirspy.io.converters`` works correctly in tests.
    """
    from nirspy.io import converters

    if direction == "nirs_to_snirf":
        return converters.nirs_to_snirf
    if direction == "snirf_to_nirs":
        return converters.snirf_to_nirs
    return None


@callback(
    Output("converter-filename", "children"),
    Output("converter-btn-convert", "disabled"),
    Input("converter-upload", "filename"),
    prevent_initial_call=True,
)
def on_file_upload(
    filename: str | None,
) -> tuple[Any, bool]:
    """Display uploaded filename and enable the convert button."""
    if not filename:
        return [], True

    badge_color = "primary"
    suffix = Path(filename).suffix.lower()
    if suffix in {".nirs", ".snirf"}:
        badge_color = "success"

    return (
        dbc.Badge(
            filename,
            color=badge_color,
            className="me-2",
        ),
        False,
    )


@callback(
    Output("converter-status", "children"),
    Output("converter-download", "data"),
    Input("converter-btn-convert", "n_clicks"),
    State("converter-upload", "contents"),
    State("converter-upload", "filename"),
    State("converter-direction", "value"),
    State("converter-strip-pii", "value"),
    prevent_initial_call=True,
)
def on_convert(
    n_clicks: int | None,
    contents: str | None,
    filename: str | None,
    direction: str,
    strip_pii: bool,
) -> tuple[Any, Any]:
    """Run the file conversion and trigger download."""
    if not n_clicks or not contents or not filename:
        raise PreventUpdate

    # Decode the base64 content from dcc.Upload
    content_string = (
        contents.split(",", 1)[1]
        if "," in contents
        else contents
    )
    raw_bytes = base64.b64decode(content_string)

    # Resolve direction config
    meta = _DIRECTION_META.get(direction)
    if meta is None:
        return render_error("Invalid conversion direction."), no_update

    expected_suffix, output_suffix = meta
    converter_fn = _get_converter(direction)
    if converter_fn is None:
        return render_error("Invalid conversion direction."), no_update

    # Validate file extension matches direction
    input_suffix = Path(filename).suffix.lower()
    if input_suffix != expected_suffix:
        return (
            render_error(
                f"File has extension {input_suffix!r}, "
                f"but direction {direction!r} expects "
                f"{expected_suffix!r}. "
                f"Please upload a {expected_suffix} file or change "
                f"the conversion direction."
            ),
            no_update,
        )

    # Write uploaded bytes to a temp file, convert, and read result
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / f"input{expected_suffix}"
            input_path.write_bytes(raw_bytes)

            stem = Path(filename).stem
            output_name = f"{stem}{output_suffix}"
            output_path = tmp_path / output_name

            converter_fn(
                input_path,
                output_path,
                overwrite=True,
                strip_pii=strip_pii,
            )

            result_bytes = output_path.read_bytes()

    except ConverterError as exc:
        logger.warning(
            "Conversion failed: %s", exc, exc_info=True
        )
        return (
            render_error(str(exc)),
            no_update,
        )
    except Exception:
        logger.exception("Unexpected error during conversion")
        return (
            render_error(
                "An unexpected error occurred during conversion. "
                "Please check that the file is valid."
            ),
            no_update,
        )

    return (
        render_error(
            f"Conversion successful! Downloading {output_name}...",
            severity="success",
        ),
        dcc.send_bytes(  # type: ignore[attr-defined,no-untyped-call]
            result_bytes,
            filename=output_name,
        ),
    )
