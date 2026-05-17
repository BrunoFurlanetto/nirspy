"""CLI entry point — ``nirspy`` command."""

from __future__ import annotations

import pathlib
import sys

import click

from nirspy import __version__


@click.group()
@click.version_option(__version__)
def main() -> None:
    """NIRSPY — NIRS Processing in Python."""


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8050, show_default=True, type=int)
@click.option("--debug/--no-debug", default=False)
def serve(host: str, port: int, debug: bool) -> None:
    """Start the Dash GUI at http://HOST:PORT."""
    raise click.ClickException("GUI not yet implemented (pre-alpha).")


@main.command()
@click.argument("pipeline_yaml", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Input SNIRF file path (overrides LoadSnirf.params.path in YAML).",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False),
    default="./results",
    show_default=True,
    help="Output directory for results.",
)
@click.option("--verbose", is_flag=True, help="Print block-by-block progress to stdout.")
def run(
    pipeline_yaml: str,
    input_path: str | None,
    output_dir: str,
    verbose: bool,
) -> None:
    """Execute a pipeline from a YAML file."""
    from nirspy.io.pipeline_runner import run_pipeline

    pipeline_p = pathlib.Path(pipeline_yaml)
    input_p = pathlib.Path(input_path) if input_path else None
    output_p = pathlib.Path(output_dir)

    result = run_pipeline(
        pipeline_path=pipeline_p,
        input_override=input_p,
        output_dir=output_p,
        verbose=verbose,
    )

    if not result.success:
        click.echo(f"Error: {result.error}", err=True)
        # Exit code 1 for validation errors, 2 for runtime errors
        code = 1 if "not found" in (result.error or "") else 2
        sys.exit(code)

    click.echo(
        f"Pipeline completed: {result.blocks_executed}/{result.total_blocks} blocks executed."
    )
    if result.output_path:
        click.echo(f"Output saved to: {result.output_path}")


if __name__ == "__main__":
    main()
