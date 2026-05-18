"""Regression tests for CondNames parsing in .nirs files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from nirspy.io.converters import _parse_nirs


def _build_minimal_mat(condnames_in_sd: bool, cond_names: list[str]) -> dict[str, object]:
    n_samples = 100
    n_conditions = len(cond_names)
    s = np.zeros((n_samples, n_conditions), dtype=np.float64)
    for i in range(n_conditions):
        s[i * 10 + 5, i] = 1.0

    sd: dict[str, object] = {
        "Lambda": np.array([760.0, 850.0]),
        "SrcPos": np.array([[0.0, 0.0, 0.0]]),
        "DetPos": np.array([[30.0, 0.0, 0.0]]),
        "MeasList": np.array([[1, 1, 1, 1], [1, 1, 1, 2]]),
        "nSrcs": 1,
        "nDets": 1,
        "SpatialUnit": "mm",
    }

    if condnames_in_sd:
        sd["CondNames"] = np.array(cond_names, dtype=object)

    mat: dict[str, object] = {
        "d": np.random.RandomState(0).rand(n_samples, 2).astype(np.float64),
        "t": np.arange(n_samples) / 10.0,
        "s": s,
        "SD": sd,
    }

    if not condnames_in_sd:
        mat["CondNames"] = np.array(cond_names, dtype=object)

    return mat


@pytest.mark.parametrize("location", ["sd", "toplevel"])
def test_condnames_picked_up_from_both_locations(tmp_path: Path, location: str) -> None:
    """Conversor deve ler CondNames de SD (HOMER2) ou top-level (HOMER3)."""
    cond_names = ["B", "D", "F", "I", "S"]
    mat = _build_minimal_mat(condnames_in_sd=(location == "sd"), cond_names=cond_names)

    path = tmp_path / "fixture.nirs"
    savemat(path, mat)

    data = _parse_nirs(path)
    names_seen = [ev.name for ev in data.stim_events]
    assert set(names_seen) == set(cond_names)
    assert "1" not in names_seen
    assert "1.0" not in names_seen


def test_no_condnames_falls_back_to_numeric(tmp_path: Path) -> None:
    """Sem CondNames em qualquer lugar, fallback usa 1, 2, 3 (sem .0)."""
    mat = _build_minimal_mat(condnames_in_sd=False, cond_names=["A", "B", "C"])
    del mat["CondNames"]

    path = tmp_path / "no_condnames.nirs"
    savemat(path, mat)

    data = _parse_nirs(path)
    names_seen = sorted({ev.name for ev in data.stim_events})
    assert names_seen == ["1", "2", "3"]
