"""CLI entry point — `nirspy` command."""

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
    """Sobe a GUI Dash em http://HOST:PORT."""
    raise click.ClickException("GUI ainda não implementada (pre-alpha).")


@main.command()
@click.argument("pipeline", type=click.Path(exists=True, dir_okay=False))
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def run(pipeline: str, input_file: str) -> None:
    """Roda PIPELINE (YAML) em INPUT_FILE em modo batch."""
    raise click.ClickException("Runner batch ainda não implementado (pre-alpha).")


if __name__ == "__main__":
    main()
