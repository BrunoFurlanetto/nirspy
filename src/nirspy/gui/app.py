"""Dash application factory.

Creates and configures the Dash app with Bootstrap theme, registers all
callbacks, and returns a ready-to-run app instance.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc

from nirspy.gui.layouts import create_layout


def create_app(*, debug: bool = False) -> dash.Dash:
    """Create and configure the Dash application.

    Parameters
    ----------
    debug:
        When *True*, enables Dash debug mode (hot-reload, verbose errors).

    Returns
    -------
    dash.Dash
        Fully configured Dash app with layout and callbacks registered.
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        title="NIRSPY",
        update_title="NIRSPY - Loading...",
    )

    app.layout = create_layout()

    # Import and register callbacks (side-effect import pattern used by Dash)
    from nirspy.gui.callbacks import pipeline_callbacks as _pc  # noqa: F401

    # Silence unused import warning — callbacks register via @app.callback
    _ = _pc

    return app
