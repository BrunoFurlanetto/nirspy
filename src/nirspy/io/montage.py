"""Montage IO helpers for SNIRF probe positions (T-026).

Read/write montage data from SNIRF files and sidecar JSON files.
Sidecar files have precedence over SNIRF-embedded positions, allowing
user-edited positions to persist across runs (D6).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for montage dicts
MontageDict = dict[str, Any]


def read_snirf_montage(path: str | Path) -> MontageDict | None:
    """Read source/detector positions directly from a SNIRF file.

    Uses h5py-direct access (same pattern as ``read_snirf_condition_names``).
    Returns ``None`` if the file is missing or lacks probe position data.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(p, "r") as f:
            if "nirs" not in f:
                return None
            nirs = f["nirs"]
            if "probe" not in nirs:
                return None
            probe = nirs["probe"]

            src_pos = _read_positions(probe, "source")
            if src_pos is None:
                return None
            det_pos = _read_positions(probe, "detector")
            if det_pos is None:
                return None

            return {"sources": src_pos, "detectors": det_pos}
    except (OSError, KeyError, ValueError):
        return None


def _read_positions(
    probe: Any, prefix: str,
) -> list[list[float]] | None:
    """Read source or detector positions from HDF5 probe group."""
    import numpy as np

    key_3d = f"{prefix}Pos3D"
    key_2d = f"{prefix}Pos2D"

    if key_3d in probe:
        pos = np.array(probe[key_3d], dtype=np.float64)
        return [[float(row[0]), float(row[1])] for row in pos]
    elif key_2d in probe:
        pos = np.array(probe[key_2d], dtype=np.float64)
        return [[float(row[0]), float(row[1])] for row in pos]
    return None


def _sidecar_path(snirf_path: str | Path) -> Path:
    """Return the sidecar JSON path for a given SNIRF file."""
    p = Path(snirf_path)
    return p.parent / f"{p.stem}.montage.json"


def load_sidecar_montage(snirf_path: str | Path) -> MontageDict | None:
    """Load montage from sidecar if it exists. Returns None otherwise."""
    sp = _sidecar_path(snirf_path)
    if not sp.exists():
        return None
    try:
        data: MontageDict = json.loads(sp.read_text(encoding="utf-8"))
        if "sources" in data and "detectors" in data:
            return data
        return None
    except (json.JSONDecodeError, OSError):
        return None


def save_sidecar_montage(
    snirf_path: str | Path, montage_dict: MontageDict,
) -> None:
    """Save montage to ``<snirf>.montage.json`` sidecar file."""
    sp = _sidecar_path(snirf_path)
    sp.write_text(
        json.dumps(montage_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved sidecar montage: %s", sp)


def resolve_montage(
    snirf_path: str | Path,
) -> tuple[MontageDict | None, str]:
    """Resolve montage with sidecar precedence (D6).

    Chain: sidecar > SNIRF native > None."""
    sidecar = load_sidecar_montage(snirf_path)
    if sidecar is not None:
        return sidecar, "sidecar"
    snirf = read_snirf_montage(snirf_path)
    if snirf is not None:
        return snirf, "snirf"
    return None, "missing"
