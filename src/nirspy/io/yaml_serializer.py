"""YAML serialiser for :class:`~nirspy.domain.pipeline.Pipeline`.

Format (schema_version 0.1)
----------------------------
::

    schema_version: "0.1"
    name: my-pipeline
    description: ""
    steps:
      - block_id: load_snirf
        enabled: true
    params:
      load_snirf:
        path: /data/subject01.snirf

Round-trip guarantee
--------------------
``load_pipeline(dump_pipeline(pipeline, path), registry)`` reconstructs an
equivalent pipeline, and a second ``dump_pipeline`` call produces **identical
bytes** to the first.  This is guaranteed by:

- Using ``yaml.dump`` with ``sort_keys=True, default_flow_style=False, allow_unicode=True``.
- All params values are plain dicts (produced by :func:`dataclasses.asdict` before
  storing in ``Pipeline.params``).  No custom YAML representers are needed.

Usage
-----
::

    from pathlib import Path
    from nirspy.blocks import registry
    from nirspy.domain.pipeline import Pipeline
    from nirspy.io.yaml_serializer import dump_pipeline, load_pipeline

    # serialise
    dump_pipeline(pipeline, Path("pipeline.yml"))

    # deserialise
    pipeline2 = load_pipeline(Path("pipeline.yml"), registry)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from nirspy.domain.exceptions import ValidationError
from nirspy.domain.pipeline import Pipeline, RegistryProtocol

_SCHEMA_VERSION = "0.1"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dump_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Serialise *pipeline* to a YAML file at *path*.

    Parameters
    ----------
    pipeline:
        The pipeline to serialise.  ``Pipeline.params`` values must be either
        plain ``dict`` objects or dataclass instances (converted via
        :func:`dataclasses.asdict`).
    path:
        Destination file.  Parent directories must exist.
    """
    data = _pipeline_to_dict(pipeline)
    yaml_text = yaml.dump(
        data,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
    )
    path.write_text(yaml_text, encoding="utf-8")


def load_pipeline(path: Path, registry: RegistryProtocol) -> Pipeline:
    """Deserialise a pipeline from the YAML file at *path*.

    Parameters
    ----------
    path:
        Source YAML file produced by :func:`dump_pipeline`.
    registry:
        Used to resolve ``block_id`` strings to :class:`~nirspy.domain.block.Block`
        instances.

    Returns
    -------
    Pipeline
        Reconstructed pipeline with blocks from *registry* and params restored.

    Raises
    ------
    ValidationError
        When ``schema_version`` is missing or incompatible.
    KeyError
        When a ``block_id`` is not found in *registry*.
    FileNotFoundError
        When *path* does not exist.
    """
    raw_text = path.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(raw_text)
    return _pipeline_from_dict(data, registry)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pipeline_to_dict(pipeline: Pipeline) -> dict[str, Any]:
    """Convert *pipeline* to a plain dict suitable for YAML serialisation."""
    serialised_params: dict[str, Any] = {}
    for block_id, params_value in pipeline.params.items():
        if dataclasses.is_dataclass(params_value) and not isinstance(params_value, type):
            serialised_params[block_id] = dataclasses.asdict(params_value)
        else:
            serialised_params[block_id] = params_value

    return {
        "description": pipeline.description,
        "name": pipeline.name,
        "params": serialised_params,
        "schema_version": _SCHEMA_VERSION,
        "steps": [
            {
                "block_id": step.spec.block_id,
                "enabled": step.spec.enabled,
            }
            for step in pipeline.steps
        ],
    }


def _pipeline_from_dict(data: dict[str, Any], registry: RegistryProtocol) -> Pipeline:
    """Reconstruct a :class:`Pipeline` from a deserialised dict."""
    version = data.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValidationError(
            f"Unsupported pipeline schema version: {version!r}. "
            f"Expected {_SCHEMA_VERSION!r}."
        )

    steps_raw: list[dict[str, Any]] = data.get("steps", [])
    steps = [registry.get(entry["block_id"]) for entry in steps_raw]

    params: dict[str, Any] = data.get("params") or {}

    # Restore params dataclasses when the block spec declares a params_class
    restored_params: dict[str, Any] = {}
    for block_id, params_dict in params.items():
        # Find the block to check its params_class
        try:
            block = registry.get(block_id)
        except KeyError:
            restored_params[block_id] = params_dict
            continue

        params_class = block.spec.params_class
        if params_class is not None and isinstance(params_dict, dict):
            try:
                restored_params[block_id] = params_class(**params_dict)
            except TypeError:
                # Unexpected keys or missing fields — store raw dict and let
                # downstream validation catch it.
                restored_params[block_id] = params_dict
        else:
            restored_params[block_id] = params_dict

    return Pipeline(
        name=data.get("name", ""),
        description=data.get("description", ""),
        steps=steps,
        params=restored_params,
    )
