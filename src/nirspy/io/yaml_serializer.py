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
- Params values are serialised via :func:`dataclasses.asdict` (from ``block.params``) and
  restored as dataclass instances during :func:`load_pipeline`.

Registry contract
-----------------
:func:`load_pipeline` expects a :class:`~nirspy.blocks.registry.BlockRegistry` (which
stores classes, not instances, per ADR-009) rather than the domain-level
``RegistryProtocol`` (which still declares ``get() -> Block``).  The serialiser
resolves the class, instantiates params, then instantiates the block.

Block class convention
----------------------
Each block class **must** expose its static :class:`~nirspy.domain.block.BlockSpec`
as a class-level attribute named ``SPEC`` (``ClassVar[BlockSpec]``).  This allows
the serialiser to read ``params_class`` without instantiating the block.

Usage
-----
::

    from pathlib import Path
    from nirspy.blocks import registry
    from nirspy.domain.pipeline import Pipeline
    from nirspy.io.yaml_serializer import dump_pipeline, load_pipeline

    # serialise
    dump_pipeline(pipeline, Path("pipeline.yml"), overwrite=True)

    # deserialise
    pipeline2 = load_pipeline(Path("pipeline.yml"), registry)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from nirspy.blocks.registry import BlockRegistry
from nirspy.domain.block import BlockSpec
from nirspy.domain.conditions import global_conditions_from_dict, global_conditions_to_dict
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.pipeline import Pipeline

_SCHEMA_VERSION = "0.1"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dump_pipeline(pipeline: Pipeline, path: Path, *, overwrite: bool = False) -> None:
    """Serialise *pipeline* to a YAML file at *path*.

    Parameters
    ----------
    pipeline:
        The pipeline to serialise.  Each block's params are extracted via
        ``block.params`` when ``block.spec.params_class`` is not ``None``.
    path:
        Destination file.  Parent directories must exist.
    overwrite:
        When *False* (default), raise :class:`FileExistsError` if *path*
        already exists.  Set to *True* to silently replace the file.

    Raises
    ------
    FileExistsError
        When *path* exists and *overwrite* is *False*.
    """
    if not overwrite and path.exists():
        raise FileExistsError(
            f"Cannot write pipeline: '{path}' already exists. "
            f"Pass overwrite=True to replace the existing file."
        )
    data = _pipeline_to_dict(pipeline)
    yaml_text = yaml.dump(
        data,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
    )
    path.write_text(yaml_text, encoding="utf-8")


def load_pipeline(path: Path, registry: BlockRegistry) -> Pipeline:
    """Deserialise a pipeline from the YAML file at *path*.

    Parameters
    ----------
    path:
        Source YAML file produced by :func:`dump_pipeline`.
    registry:
        :class:`~nirspy.blocks.registry.BlockRegistry` used to resolve
        ``block_id`` strings to block **classes**, which are then instantiated
        with the params read from the YAML file (ADR-009).

    Returns
    -------
    Pipeline
        Reconstructed pipeline with fully instantiated blocks.

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


def _get_spec_from_class(block_cls: type[Any]) -> BlockSpec | None:
    """Return the :class:`~nirspy.domain.block.BlockSpec` for *block_cls* if available.

    Looks for the ``SPEC`` class attribute (the conventional name used by all
    built-in blocks).  Returns ``None`` when the attribute is absent so callers
    can degrade gracefully.
    """
    spec = getattr(block_cls, "SPEC", None)
    if isinstance(spec, BlockSpec):
        return spec
    return None


def _pipeline_to_dict(pipeline: Pipeline) -> dict[str, Any]:
    """Convert *pipeline* to a plain dict suitable for YAML serialisation.

    Params are extracted directly from each block's ``params`` attribute when
    the block's ``spec.params_class`` is set.  If a block has no params the
    entry is omitted from the ``params`` mapping.
    """
    serialised_params: dict[str, Any] = {}
    for step in pipeline.steps:
        block_spec = step.spec
        if block_spec.params_class is not None and hasattr(step, "params"):
            params_value = step.params
            if dataclasses.is_dataclass(params_value) and not isinstance(params_value, type):
                serialised_params[block_spec.block_id] = dataclasses.asdict(params_value)
            else:
                serialised_params[block_spec.block_id] = params_value

    d: dict[str, Any] = {
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
    if pipeline.global_conditions is not None:
        d["global_conditions"] = global_conditions_to_dict(pipeline.global_conditions)
    return d


def _pipeline_from_dict(data: dict[str, Any], registry: BlockRegistry) -> Pipeline:
    """Reconstruct a :class:`Pipeline` from a deserialised dict.

    For each step the serialiser:

    1. Resolves the block **class** via ``registry.get(block_id)``.
    2. Reads the raw params dict from the YAML ``params`` section.
    3. Looks up ``params_class`` from the class-level ``SPEC`` attribute.
    4. Instantiates a params dataclass via ``params_class(**raw_params)`` when
       both ``params_class`` and ``raw_params`` are available.
    5. Instantiates the block via ``block_cls(params_obj)`` (or ``block_cls()``
       when no params are required).
    """
    version = data.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValidationError(
            f"Unsupported pipeline schema version: {version!r}. "
            f"Expected {_SCHEMA_VERSION!r}."
        )

    steps_raw: list[dict[str, Any]] = data.get("steps", [])
    raw_params_map: dict[str, Any] = data.get("params") or {}

    steps = []
    for entry in steps_raw:
        block_id: str = entry["block_id"]
        block_cls = registry.get(block_id)  # type[Block]

        # Resolve params_class from the class-level SPEC attribute (convention).
        spec = _get_spec_from_class(block_cls)
        params_class = spec.params_class if spec is not None else None

        raw_params: Any = raw_params_map.get(block_id)

        if params_class is not None and isinstance(raw_params, dict):
            try:
                params_obj = params_class(**raw_params)
            except TypeError:
                # Unexpected keys or missing fields — store raw dict and let
                # downstream validation surface the issue.
                params_obj = raw_params
            block = block_cls(params_obj)  # type: ignore[call-arg]
        elif params_class is not None and raw_params is not None:
            # raw_params may already be a suitable object — pass it through.
            block = block_cls(raw_params)  # type: ignore[call-arg]
        else:
            # Block declares no params — instantiate with no arguments.
            block = block_cls()

        steps.append(block)

    gc_data = data.get("global_conditions")
    global_conditions = global_conditions_from_dict(gc_data) if gc_data else None

    return Pipeline(
        name=data.get("name", ""),
        description=data.get("description", ""),
        steps=steps,
        global_conditions=global_conditions,
    )
