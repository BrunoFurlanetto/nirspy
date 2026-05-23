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
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

from nirspy.domain.exceptions import ConverterError
from nirspy.engine.exceptions import get_user_message
from nirspy.gui.components.error_display import render_error

logger = logging.getLogger(__name__)

# Map direction -> (expected input suffix, output suffix)
_DIRECTION_META: dict[str, tuple[str, str]] = {
    "nirs_to_snirf": (".nirs", ".snirf"),
    "snirf_to_nirs": (".snirf", ".nirs"),
    "oxysoft_txt_to_snirf": (".txt", ".snirf"),
}


def _get_converter(direction: str) -> Any:
    """Resolve the converter function for a given direction.

    Imported lazily so that ``unittest.mock.patch`` on the module-level
    names in ``nirspy.io.converters`` works correctly in tests.
    """
    from nirspy.io import converters, oxysoft_txt

    if direction == "nirs_to_snirf":
        return converters.nirs_to_snirf
    if direction == "snirf_to_nirs":
        return converters.snirf_to_nirs
    if direction == "oxysoft_txt_to_snirf":
        return oxysoft_txt.oxysoft_txt_to_snirf
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


def _build_distance_info(
    stats: dict[str, float | int],
) -> tuple[Any, str]:
    """Return (info-children, suggested-scale-value) for the panel."""
    mean_d = float(stats.get("mean", 0.0))
    min_d = float(stats.get("min", 0.0))
    max_d = float(stats.get("max", 0.0))
    n_ch = int(stats.get("n_channels", 0))

    # Heuristic: typical adult fNIRS S-D distance is 25-40 mm. If the
    # mean is suspiciously small (< 10) we suggest a x10 scale because
    # the .nirs file is likely in cm.
    suggested_scale = "10" if 0.0 < mean_d < 10.0 else "1"

    if suggested_scale != "1":
        suspicion_msg = html.Div(
            [
                html.Strong("Possible unit mismatch detected. "),
                "The mean S-D distance looks like centimeters. "
                "Apply x10 to convert to millimeters before writing "
                "the SNIRF file.",
            ],
            className="small text-warning mb-2",
        )
    else:
        suspicion_msg = html.Div(
            "Confirm the source-detector distance is correct before "
            "converting. Adjust the scale factor if needed.",
            className="small text-muted mb-2",
        )

    info = [
        html.Div(
            [
                html.I(className="bi bi-rulers me-2"),
                html.Strong("Probe distance check"),
            ],
            className="mb-2",
        ),
        html.Div(
            f"Detected mean S-D distance: {mean_d:.3f} "
            f"(min {min_d:.3f}, max {max_d:.3f}) across "
            f"{n_ch} channel pair(s).",
            className="small mb-2",
        ),
        suspicion_msg,
    ]
    return info, suggested_scale


@callback(
    Output("converter-probe-distance-info", "children"),
    Output("converter-probe-distance-panel", "style"),
    Output("converter-pos-scale", "value"),
    Input("converter-upload", "contents"),
    Input("converter-upload", "filename"),
    Input("converter-direction", "value"),
    prevent_initial_call=True,
)
def on_inspect_probe_distance(
    contents: str | None,
    filename: str | None,
    direction: str | None,
) -> tuple[Any, dict[str, str], Any]:
    """Inspect the uploaded .nirs file and show the probe-distance panel.

    Only runs for the ``nirs_to_snirf`` direction with a .nirs file —
    other combinations hide the panel and reset the scale to x1.
    """
    hidden = {"display": "none"}
    if not contents or not filename or direction != "nirs_to_snirf":
        return [], hidden, "1"
    if Path(filename).suffix.lower() != ".nirs":
        return [], hidden, "1"

    content_string = (
        contents.split(",", 1)[1]
        if "," in contents
        else contents
    )
    try:
        raw_bytes = base64.b64decode(content_string)
    except (ValueError, TypeError):
        return [], hidden, "1"

    from nirspy.io.converters import inspect_nirs_distances

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_in = Path(tmp_dir) / "input.nirs"
            tmp_in.write_bytes(raw_bytes)
            stats = inspect_nirs_distances(tmp_in)
    except ConverterError as exc:
        logger.warning("Probe-distance inspection failed: %s", exc)
        return [], hidden, "1"
    except Exception:
        logger.exception("Unexpected error inspecting probe distance")
        return [], hidden, "1"

    info, suggested = _build_distance_info(stats)
    return info, {"display": "block"}, suggested


@callback(
    Output("converter-status", "children"),
    Output("converter-download", "data"),
    Input("converter-btn-convert", "n_clicks"),
    State("converter-upload", "contents"),
    State("converter-upload", "filename"),
    State("converter-direction", "value"),
    State("converter-strip-pii", "value"),
    State("converter-pos-scale", "value"),
    prevent_initial_call=True,
)
def on_convert(
    n_clicks: int | None,
    contents: str | None,
    filename: str | None,
    direction: str,
    strip_pii: bool,
    pos_scale: str | None,
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

            extra_kwargs: dict[str, Any] = {}
            if direction == "nirs_to_snirf":
                try:
                    scale_val = float(pos_scale) if pos_scale else 1.0
                except (TypeError, ValueError):
                    scale_val = 1.0
                extra_kwargs["pos_scale"] = scale_val

            converter_fn(
                input_path,
                output_path,
                overwrite=True,
                strip_pii=strip_pii,
                **extra_kwargs,
            )

            result_bytes = output_path.read_bytes()

    except ConverterError as exc:
        logger.warning(
            "Conversion failed: %s", exc, exc_info=True
        )
        return (
            render_error(get_user_message(exc)),
            no_update,
        )
    except Exception:
        logger.exception("Unexpected error during conversion")
        return (
            render_error(
                "An unexpected error occurred. "
                "Please check the log file for details."
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
