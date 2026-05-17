"""Pipeline callbacks — placeholder for 5B.

This module is imported by ``create_app()`` to ensure callback registration.
In 5A (foundation), it contains no callbacks — just a marker constant so the
import succeeds cleanly.
"""

from __future__ import annotations

# Callbacks will be added in Etapa 5B (pipeline builder).
# This module exists so that app.py can import it without error.
REGISTERED: bool = True
