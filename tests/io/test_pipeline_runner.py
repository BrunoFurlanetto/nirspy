"""Tests for nirspy.io.pipeline_runner — standalone runner logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nirspy.io.pipeline_runner import RunResult, run_pipeline


class TestRunPipelineValidation:
    """Tests for input validation in run_pipeline."""

    def test_nonexistent_pipeline_file(self, tmp_path: Path) -> None:
        result = run_pipeline(tmp_path / "nonexistent.yml")
        assert not result.success
        assert "not found" in (result.error or "")

    def test_invalid_yaml_content(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yml"
        bad_yaml.write_text("not: a: valid: pipeline:", encoding="utf-8")
        result = run_pipeline(bad_yaml)
        assert not result.success
        assert result.error is not None

    def test_nonexistent_input_override(self, tmp_path: Path) -> None:
        """When --input points to a file that doesn't exist."""
        pipeline_yaml = tmp_path / "pipe.yml"
        # Write a valid-looking YAML that would load
        pipeline_yaml.write_text(
            'schema_version: "0.1"\nname: t\ndescription: t\nsteps: []\nparams: {}\n',
            encoding="utf-8",
        )
        result = run_pipeline(
            pipeline_yaml,
            input_override=tmp_path / "missing.snirf",
        )
        assert not result.success
        assert "not found" in (result.error or "")


class TestRunPipelineExecution:
    """Tests for the execution path with mocked pipeline loading."""

    def test_empty_pipeline_succeeds(self, tmp_path: Path) -> None:
        """A pipeline with no steps should succeed with 0 blocks executed."""
        pipeline_yaml = tmp_path / "empty.yml"
        pipeline_yaml.write_text(
            'schema_version: "0.1"\nname: empty\ndescription: ""\nsteps: []\nparams: {}\n',
            encoding="utf-8",
        )
        result = run_pipeline(pipeline_yaml, output_dir=tmp_path / "out")
        assert result.success
        assert result.blocks_executed == 0

    def test_execution_error_returns_failure(self, tmp_path: Path) -> None:
        """ExecutionError during run_pipeline_sync is caught."""
        from nirspy.domain.exceptions import ExecutionError
        from nirspy.domain.pipeline import Pipeline

        pipeline_yaml = tmp_path / "pipe.yml"
        pipeline_yaml.write_text(
            'schema_version: "0.1"\nname: t\ndescription: ""\nsteps: []\nparams: {}\n',
            encoding="utf-8",
        )

        # Mock load_pipeline to return a pipeline with a block that will fail
        fake_pipeline = Pipeline(name="t", description="", steps=[])

        with patch(
            "nirspy.io.pipeline_runner.load_pipeline", return_value=fake_pipeline
        ), patch(
            "nirspy.io.pipeline_runner.run_pipeline_sync",
            side_effect=ExecutionError("block X failed"),
        ):
            result = run_pipeline(pipeline_yaml, output_dir=tmp_path / "out")

        assert not result.success
        assert "block X failed" in (result.error or "")


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_defaults(self) -> None:
        r = RunResult(success=True, output_path=None)
        assert r.blocks_executed == 0
        assert r.total_blocks == 0
        assert r.error is None
