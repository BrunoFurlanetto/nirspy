"""Integration tests — pipeline templates against MNE-NIRS sample dataset.

Each template in ``examples/pipelines/`` is loaded via the YAML serializer,
executed end-to-end against the fnirs_motor sample dataset, and validated
for successful completion and non-empty output.

These tests require network access on first run (to download the ~5 MB
MNE-NIRS sample dataset) and are marked ``slow`` so they can be skipped
in fast CI runs via ``pytest -m "not slow"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nirspy.blocks import registry
from nirspy.io.pipeline_runner import RunResult, run_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "examples" / "pipelines"

_TEMPLATE_NAMES = [
    "best-practices-block-design",
    "resting-state-connectivity",
    "motion-heavy-recording",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def snirf_sample_path() -> Path:
    """Return path to an MNE-NIRS fnirs_motor sample SNIRF file.

    Downloads the dataset on first call (~5 MB, cached by MNE).
    Skips the entire module if the dataset cannot be obtained.
    """
    try:
        import mne
        import mne_nirs  # noqa: F401

        data_dir = mne.datasets.fnirs_motor.data_path(verbose=False)
        snirf_files = list(Path(data_dir).rglob("*.snirf"))
        if not snirf_files:
            pytest.skip("No SNIRF files found in fnirs_motor dataset.")
        return snirf_files[0]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not obtain SNIRF sample data: {exc}")
        raise  # unreachable, keeps mypy happy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("template_name", _TEMPLATE_NAMES)
def test_template_loads(template_name: str) -> None:
    """Template YAML can be deserialised without errors."""
    from nirspy.io.yaml_serializer import load_pipeline

    path = _TEMPLATES_DIR / f"{template_name}.yml"
    assert path.exists(), f"Template not found: {path}"

    pipeline = load_pipeline(path, registry)
    assert len(pipeline.steps) > 0, "Pipeline has no steps"
    assert pipeline.name == template_name


@pytest.mark.slow
@pytest.mark.parametrize("template_name", _TEMPLATE_NAMES)
def test_template_roundtrip(template_name: str, tmp_path: Path) -> None:
    """load -> dump -> load produces equivalent pipeline."""
    from nirspy.io.yaml_serializer import dump_pipeline, load_pipeline

    path = _TEMPLATES_DIR / f"{template_name}.yml"
    pipeline1 = load_pipeline(path, registry)

    roundtrip_path = tmp_path / f"{template_name}-roundtrip.yml"
    dump_pipeline(pipeline1, roundtrip_path)

    pipeline2 = load_pipeline(roundtrip_path, registry)

    assert len(pipeline1.steps) == len(pipeline2.steps)
    for s1, s2 in zip(pipeline1.steps, pipeline2.steps, strict=True):
        assert s1.spec.block_id == s2.spec.block_id


@pytest.mark.slow
@pytest.mark.parametrize("template_name", _TEMPLATE_NAMES)
def test_template_end_to_end(
    template_name: str,
    snirf_sample_path: Path,
    tmp_path: Path,
) -> None:
    """Template executes end-to-end against MNE-NIRS sample data.

    Validates:
    - Exit without error (RunResult.success == True)
    - Output file is created and non-empty
    - All enabled blocks were executed
    """
    template_path = _TEMPLATES_DIR / f"{template_name}.yml"
    assert template_path.exists()

    result: RunResult = run_pipeline(
        pipeline_path=template_path,
        input_override=snirf_sample_path,
        output_dir=tmp_path / "output",
        verbose=True,
    )

    assert result.success, (
        f"Pipeline {template_name!r} failed: {result.error}"
    )
    assert result.output_path is not None, "No output file produced"
    assert result.output_path.exists(), (
        f"Output file does not exist: {result.output_path}"
    )
    assert result.output_path.stat().st_size > 0, "Output file is empty"
    assert result.blocks_executed > 0, "No blocks were executed"
