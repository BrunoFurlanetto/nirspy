"""Error display component -- user-friendly error alerts.

Wraps error messages in ``dbc.Alert`` with configurable severity.
No stack traces are exposed to the user (security best practice).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc


def render_error(
    message: str,
    *,
    severity: str = "danger",
) -> dbc.Alert:
    """Render an error message as a Bootstrap alert.

    Parameters
    ----------
    message:
        Human-readable error description.  Must NOT contain
        stack traces or internal paths.
    severity:
        Bootstrap alert colour (``"danger"``, ``"warning"``, ``"info"``).
        Defaults to ``"danger"`` (red).

    Returns
    -------
    dbc.Alert
        Dismissable alert component.
    """
    return dbc.Alert(
        message,
        color=severity,
        dismissable=True,
        is_open=True,
        className="mt-2",
    )
