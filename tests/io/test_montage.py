"""Tests for montage IO and probe head silhouette (T-026)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


class TestMontageIO:
    """Tests for io/montage.py functions."""

    def test_sidecar_round_trip(self, tmp_path: Path) -> None:
        from nirspy.io.montage import (
            load_sidecar_montage,
            save_sidecar_montage,
        )

        snirf_path = tmp_path / 'test.snirf'
        snirf_path.write_text("")  # dummy file
        montage = {
            "sources": [[0.0, 0.1], [0.2, 0.3]],
            "detectors": [[0.4, 0.5]],
        }
        save_sidecar_montage(snirf_path, montage)

        loaded = load_sidecar_montage(snirf_path)
        assert loaded is not None
        assert loaded["sources"] == [[0.0, 0.1], [0.2, 0.3]]
        assert loaded["detectors"] == [[0.4, 0.5]]

    def test_sidecar_missing_returns_none(self, tmp_path: Path) -> None:
        from nirspy.io.montage import load_sidecar_montage

        snirf_path = tmp_path / 'nonexistent.snirf'
        assert load_sidecar_montage(snirf_path) is None

    def test_resolve_missing(self, tmp_path: Path) -> None:
        from nirspy.io.montage import resolve_montage

        snirf_path = tmp_path / 'no.snirf'
        montage, source = resolve_montage(snirf_path)
        assert montage is None
        assert source == "missing"

    def test_resolve_sidecar_precedence(self, tmp_path: Path) -> None:
        from nirspy.io.montage import resolve_montage, save_sidecar_montage

        snirf_path = tmp_path / 'test.snirf'
        snirf_path.write_text("")
        montage_data = {
            "sources": [[1.0, 2.0]],
            "detectors": [[3.0, 4.0]],
        }
        save_sidecar_montage(snirf_path, montage_data)

        montage, source = resolve_montage(snirf_path)
        assert montage is not None
        assert source == "sidecar"
        assert montage["sources"] == [[1.0, 2.0]]

    def test_read_snirf_montage_no_file(self) -> None:
        from nirspy.io.montage import read_snirf_montage

        assert read_snirf_montage("/nonexistent/path.snirf") is None

    def test_sidecar_invalid_json(self, tmp_path: Path) -> None:
        from nirspy.io.montage import load_sidecar_montage

        snirf_path = tmp_path / 'test.snirf'
        snirf_path.write_text("")
        sidecar = tmp_path / 'test.montage.json'
        sidecar.write_text("not valid json")
        assert load_sidecar_montage(snirf_path) is None


class TestProbeHeadSilhouette:
    """Tests for head silhouette in probe_viewer."""

    def test_head_traces_present(self) -> None:
        import plotly.graph_objects as go
        from nirspy.gui.components.probe_viewer import _draw_head_silhouette

        fig = go.Figure()
        _draw_head_silhouette(fig, 0.05)
        # Should have at least 4 traces: head, nose, left ear, right ear
        assert len(fig.data) >= 4

    def test_head_traces_zero_scale_fallback(self) -> None:
        import plotly.graph_objects as go
        from nirspy.gui.components.probe_viewer import _draw_head_silhouette

        fig = go.Figure()
        _draw_head_silhouette(fig, 0.0)
        assert len(fig.data) >= 4  # should not crash

    def test_render_no_info_shows_fallback(self) -> None:
        from nirspy.gui.components.probe_viewer import render_probe_viewer

        result = render_probe_viewer(None)
        assert result.id == "probe-viewer-content"
