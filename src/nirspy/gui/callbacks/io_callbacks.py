"""IO callbacks -- save pipeline as YAML download, load pipeline from YAML upload."""

from __future__ import annotations

import base64
import dataclasses
import uuid
from typing import Any

import yaml
from dash import Input, Output, State, callback, dcc, no_update

from nirspy.blocks import registry
from nirspy.domain.block import BlockSpec

_SCHEMA_VERSION = "0.1"


@callback(
    Output("download-pipeline", "data"),
    Input("btn-save-pipeline", "n_clicks"),
    State("pipeline-state", "data"),
    prevent_initial_call=True,
)
def save_pipeline(n_clicks: int | None, pipeline_state: list[dict[str, Any]]) -> Any:
    """Serialize pipeline state to YAML and trigger download."""
    if not n_clicks or not pipeline_state:
        return no_update

    # Build YAML-compatible dict matching the schema used by yaml_serializer
    params_map: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    for entry in pipeline_state:
        block_id = entry["block_id"]
        steps.append({
            "block_id": block_id,
            "enabled": entry.get("enabled", True),
        })
        p = entry.get("params", {})
        if p:
            params_map[block_id] = p

    data = {
        "description": "",
        "name": "untitled-pipeline",
        "params": params_map,
        "schema_version": _SCHEMA_VERSION,
        "steps": steps,
    }

    yaml_text = yaml.dump(
        data,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
    )

    return dcc.send_string(yaml_text, filename="pipeline.yml")  # type: ignore[attr-defined,no-untyped-call]


@callback(
    Output("pipeline-state", "data", allow_duplicate=True),
    Input("upload-pipeline", "contents"),
    State("upload-pipeline", "filename"),
    prevent_initial_call=True,
)
def load_pipeline_from_upload(
    contents: str | None,
    filename: str | None,
) -> Any:
    """Parse uploaded YAML and populate the pipeline state."""
    if not contents:
        return no_update

    # dcc.Upload sends content as "data:mime;base64,ENCODED"
    content_string = contents.split(",", 1)[1] if "," in contents else contents
    decoded = base64.b64decode(content_string).decode("utf-8")
    data: dict[str, Any] = yaml.safe_load(decoded)

    if not isinstance(data, dict):
        return no_update

    steps_raw: list[dict[str, Any]] = data.get("steps", [])
    raw_params_map: dict[str, Any] = data.get("params") or {}

    pipeline_state: list[dict[str, Any]] = []
    for step in steps_raw:
        block_id: str = step["block_id"]

        # Verify block exists in registry
        try:
            block_cls = registry.get(block_id)
        except KeyError:
            continue

        spec: BlockSpec = block_cls.SPEC  # type: ignore[attr-defined]

        # Resolve params: from YAML or defaults
        raw_params: dict[str, Any] = raw_params_map.get(block_id, {})
        if (not raw_params and spec.params_class is not None
                and dataclasses.is_dataclass(spec.params_class)):
                try:
                    default_obj = spec.params_class()
                    raw_params = dataclasses.asdict(default_obj)
                except TypeError:
                    raw_params = {}

        pipeline_state.append({
            "block_id": block_id,
            "instance_id": str(uuid.uuid4()),
            "params": raw_params,
            "enabled": step.get("enabled", True),
        })

    return pipeline_state
