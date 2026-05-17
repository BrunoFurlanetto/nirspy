"""Tests for the ``nirspy run`` CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import click.testing
import pytest

from nirspy.cli.main import main
from nirspy.io.pipeline_runner import RunResult

_PATCH_TARGET = "nirspy.io.pipeline_runner.run_pipeline"


@pytest.fixture()
def cli_runner() -> click.testing.CliRunner:
    """Click CliRunner instance."""
    return click.testing.CliRunner()


class TestRunCommandHelp:
    """Basic CLI smoke tests."""

    def test_run_help(self, cli_runner: click.testing.CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "Execute a pipeline" in result.output

    def test_run_no_args_shows_error(self, cli_runner: click.testing.CliRunner) -> None:
        result = cli_runner.invoke(main, ["run"])
        assert result.exit_code != 0


class TestRunCommandExecution:
    """Tests that exercise the run command with mocked runner."""

    def test_run_success(
        self, cli_runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        pipeline_file = tmp_path / "test.yml"
        pipeline_file.write_text("fake", encoding="utf-8")

        mock_result = RunResult(
            success=True,
            output_path=tmp_path / "results" / "result_raw.fif",
            blocks_executed=3,
            total_blocks=3,
        )

        with patch(_PATCH_TARGET, return_value=mock_result):
            result = cli_runner.invoke(main, ["run", str(pipeline_file)])

        assert result.exit_code == 0
        assert "3/3 blocks executed" in result.output

    def test_run_with_verbose(
        self, cli_runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        pipeline_file = tmp_path / "test.yml"
        pipeline_file.write_text("fake", encoding="utf-8")

        mock_result = RunResult(
            success=True,
            output_path=tmp_path / "out" / "result_raw.fif",
            blocks_executed=2,
            total_blocks=2,
        )

        with patch(_PATCH_TARGET, return_value=mock_result):
            result = cli_runner.invoke(main, ["run", str(pipeline_file), "--verbose"])

        assert result.exit_code == 0

    def test_run_failure_exits_nonzero(
        self, cli_runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        pipeline_file = tmp_path / "test.yml"
        pipeline_file.write_text("fake", encoding="utf-8")

        mock_result = RunResult(
            success=False,
            output_path=None,
            error="Input file not found: bad.snirf",
        )

        with patch(_PATCH_TARGET, return_value=mock_result):
            result = cli_runner.invoke(main, ["run", str(pipeline_file)])

        assert result.exit_code != 0

    def test_run_nonexistent_pipeline_shows_error(
        self, cli_runner: click.testing.CliRunner
    ) -> None:
        result = cli_runner.invoke(main, ["run", "nonexistent.yml"])
        assert result.exit_code != 0

    def test_run_with_output_dir(
        self, cli_runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        pipeline_file = tmp_path / "test.yml"
        pipeline_file.write_text("fake", encoding="utf-8")
        output = tmp_path / "custom_output"

        mock_result = RunResult(
            success=True,
            output_path=output / "result_raw.fif",
            blocks_executed=1,
            total_blocks=1,
        )

        with patch(_PATCH_TARGET, return_value=mock_result) as mock_rp:
            result = cli_runner.invoke(
                main, ["run", str(pipeline_file), "--output", str(output)]
            )

        assert result.exit_code == 0
        # Verify output_dir was passed correctly
        call_kwargs = mock_rp.call_args[1]
        assert call_kwargs["output_dir"] == output

    def test_run_with_input_override(
        self, cli_runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        pipeline_file = tmp_path / "test.yml"
        pipeline_file.write_text("fake", encoding="utf-8")
        input_file = tmp_path / "data.snirf"
        input_file.write_text("fake snirf", encoding="utf-8")

        mock_result = RunResult(
            success=True,
            output_path=tmp_path / "results" / "result_raw.fif",
            blocks_executed=2,
            total_blocks=2,
        )

        with patch(_PATCH_TARGET, return_value=mock_result) as mock_rp:
            result = cli_runner.invoke(
                main, ["run", str(pipeline_file), "--input", str(input_file)]
            )

        assert result.exit_code == 0
        call_kwargs = mock_rp.call_args[1]
        assert call_kwargs["input_override"] == input_file
