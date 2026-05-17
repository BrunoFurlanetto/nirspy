"""Smoke tests for GUI app factory and layout (T-006 5A).

These tests verify:
- App factory creates a Dash app without errors
- Layout renders with expected structure
- CLI serve command is wired correctly
"""

from __future__ import annotations


class TestAppFactory:
    """Tests for nirspy.gui.app.create_app."""

    def test_create_app_returns_dash_instance(self) -> None:
        from nirspy.gui.app import create_app

        app = create_app()
        assert app is not None
        assert hasattr(app, "layout")
        assert app.layout is not None

    def test_create_app_title(self) -> None:
        from nirspy.gui.app import create_app

        app = create_app()
        assert app.title == "NIRSPY"

    def test_create_app_debug_mode(self) -> None:
        from nirspy.gui.app import create_app

        app = create_app(debug=True)
        assert app is not None

    def test_layout_has_stores(self) -> None:
        from nirspy.gui.app import create_app

        app = create_app()
        layout_str = str(app.layout)
        assert "pipeline-state" in layout_str
        assert "selected-block" in layout_str

    def test_layout_has_catalog_placeholder(self) -> None:
        from nirspy.gui.app import create_app

        app = create_app()
        layout_str = str(app.layout)
        assert "block-catalog" in layout_str

    def test_layout_has_pipeline_view(self) -> None:
        from nirspy.gui.app import create_app

        app = create_app()
        layout_str = str(app.layout)
        assert "pipeline-view" in layout_str

    def test_layout_has_param_editor(self) -> None:
        from nirspy.gui.app import create_app

        app = create_app()
        layout_str = str(app.layout)
        assert "param-editor" in layout_str


class TestCLIServe:
    """Tests for the CLI serve command wiring."""

    def test_serve_command_exists(self) -> None:
        from click.testing import CliRunner

        from nirspy.cli.main import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "host" in result.output.lower()

    def test_serve_imports_create_app(self) -> None:
        """Serve command can import create_app (no import error)."""
        from nirspy.gui import create_app

        app = create_app()
        assert hasattr(app, "run")
