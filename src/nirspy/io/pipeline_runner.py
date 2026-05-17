"""PipelineRunner — standalone function to execute a pipeline from disk.

Orchestrates: load YAML -> instantiate blocks -> execute -> save output.
Does not import Click or any GUI code — testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nirspy.blocks import registry
from nirspy.domain.block import BlockResult
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ExecutionError, ValidationError
from nirspy.domain.execution import ExecutionContext, run_pipeline_sync
from nirspy.io.yaml_serializer import load_pipeline

_DEFAULT_OUTPUT_DIR = Path("./results")


@dataclass
class RunResult:
    """Outcome of a pipeline execution."""

    success: bool
    output_path: Path | None
    error: str | None = None
    blocks_executed: int = 0
    total_blocks: int = 0


def run_pipeline(
    pipeline_path: Path,
    input_override: Path | None = None,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    verbose: bool = False,
) -> RunResult:
    """Execute a pipeline YAML file end-to-end.

    Parameters
    ----------
    pipeline_path:
        Path to the pipeline YAML file.
    input_override:
        If provided, overrides the ``path`` param of the first LoadSnirf block.
    output_dir:
        Directory where results are saved. Created if absent.
    verbose:
        Print block-by-block progress to stdout.

    Returns
    -------
    RunResult
        Outcome with success flag, output path, and error details.
    """
    # --- Validate pipeline file ---
    if not pipeline_path.exists():
        return RunResult(
            success=False,
            output_path=None,
            error=f"Pipeline file not found: {pipeline_path}",
        )

    # --- Load pipeline ---
    try:
        pipeline = load_pipeline(pipeline_path, registry)
    except (ValidationError, KeyError, FileNotFoundError, Exception) as exc:  # noqa: BLE001
        return RunResult(
            success=False,
            output_path=None,
            error=f"Failed to load pipeline: {exc}",
        )

    # --- Override input path if requested ---
    if input_override is not None:
        if not input_override.exists():
            return RunResult(
                success=False,
                output_path=None,
                error=f"Input file not found: {input_override}",
            )
        _override_load_path(pipeline, input_override)

    # --- Count enabled blocks ---
    enabled_steps = [s for s in pipeline.steps if s.spec.enabled]
    total_blocks = len(enabled_steps)

    # --- Build execution context ---
    blocks_executed = 0

    def _progress(block_id: str, step: int, total: int) -> None:
        nonlocal blocks_executed
        blocks_executed = step
        if verbose:
            print(f"[{step}/{total}] {block_id}")  # noqa: T201

    context = ExecutionContext(progress=_progress)

    # --- Execute ---
    try:
        results = run_pipeline_sync(pipeline, context)
    except ExecutionError as exc:
        return RunResult(
            success=False,
            output_path=None,
            error=str(exc),
            blocks_executed=blocks_executed,
            total_blocks=total_blocks,
        )

    # --- Save output ---
    if not results:
        return RunResult(
            success=True,
            output_path=None,
            blocks_executed=0,
            total_blocks=total_blocks,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    last_result = results[-1]
    saved_path = _save_result(last_result, output_dir)

    return RunResult(
        success=True,
        output_path=saved_path,
        blocks_executed=blocks_executed,
        total_blocks=total_blocks,
    )


def _override_load_path(pipeline: Any, input_path: Path) -> None:
    """Override the path param of the first LoadSnirf block in the pipeline."""
    from nirspy.blocks.load import LoadSnirfBlock, LoadSnirfParams

    for step in pipeline.steps:
        if isinstance(step, LoadSnirfBlock):
            step.params = LoadSnirfParams(path=str(input_path))
            return


def _save_result(result: BlockResult, output_dir: Path) -> Path:
    """Save the final block result to disk based on its data type.

    Returns the path where the result was saved.
    """
    import mne

    data = result.data
    output_type = _detect_output_type(data)

    if output_type == DataType.EVOKED:
        out_path = output_dir / "result-ave.fif"
        if isinstance(data, list):
            mne.write_evokeds(str(out_path), data, overwrite=True)
        else:
            mne.write_evokeds(str(out_path), [data], overwrite=True)
        return out_path

    if output_type in (DataType.RAW, DataType.RAW_OD, DataType.RAW_HAEMO):
        out_path = output_dir / "result_raw.fif"
        data.save(str(out_path), overwrite=True)
        return out_path

    if output_type == DataType.EPOCHS:
        out_path = output_dir / "result-epo.fif"
        data.save(str(out_path), overwrite=True)
        return out_path

    # Fallback: pickle
    import pickle

    out_path = output_dir / "result.pkl"
    out_path.write_bytes(pickle.dumps(data))
    return out_path


def _detect_output_type(data: Any) -> DataType:
    """Infer DataType from the actual MNE object type."""
    import mne

    if isinstance(data, mne.Evoked):
        return DataType.EVOKED
    if isinstance(data, list) and data and isinstance(data[0], mne.Evoked):
        return DataType.EVOKED
    if isinstance(data, mne.Epochs):
        return DataType.EPOCHS
    if isinstance(data, mne.io.BaseRaw):
        ch_types = set(data.get_channel_types())
        if ch_types & {"hbo", "hbr"}:
            return DataType.RAW_HAEMO
        return DataType.RAW
    return DataType.RAW
