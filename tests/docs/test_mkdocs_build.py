"""Tests for mkdocs documentation build (T-019)."""

from __future__ import annotations

import subprocess
import sys

import pytest


class TestMkdocsBuild:
    """Verify that mkdocs build succeeds without errors."""

    @pytest.mark.slow()
    def test_mkdocs_build_succeeds(self) -> None:
        """mkdocs build --strict exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "--strict"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"mkdocs build failed:\n{result.stderr}"
        )

    @pytest.mark.slow()
    def test_no_broken_links(self) -> None:
        """mkdocs build --strict does not emit broken-link warnings."""
        result = subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "--strict"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # --strict makes warnings into errors, so returncode != 0 catches them
        assert result.returncode == 0, (
            f"mkdocs build had warnings (broken links?):\n{result.stderr}"
        )
        # Double-check: no "WARNING" in output
        assert "WARNING" not in result.stderr, (
            f"mkdocs build emitted warnings:\n{result.stderr}"
        )
