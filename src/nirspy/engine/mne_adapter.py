"""MNEAdapter — thin wrapper around MNE-NIRS for Etapa 1.

Only ``load_snirf`` is implemented here. Additional methods (OD conversion,
Beer-Lambert, etc.) are deferred to Etapa 2.
"""

from __future__ import annotations

from pathlib import Path

import mne
import mne.io

from nirspy.engine.exceptions import SnirfLoadError


class RawWrapper:
    """Lightweight container that pairs a :class:`mne.io.BaseRaw` with its source path.

    Blocks in ``nirspy.blocks`` receive this wrapper so they can access both
    the MNE object and provenance information without additional arguments.
    """

    def __init__(self, raw: mne.io.BaseRaw, source_path: Path) -> None:
        self.raw = raw
        self.source_path = source_path

    def __repr__(self) -> str:
        return f"RawWrapper(path={self.source_path!r}, n_channels={len(self.raw.ch_names)})"


class MNEAdapter:
    """Facade over MNE / MNE-NIRS I/O and signal-processing operations.

    All methods are stateless — the class holds no mutable state and can be
    instantiated once and reused across pipeline runs.
    """

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_snirf(self, path: Path) -> mne.io.BaseRaw:
        """Load a SNIRF file and return a :class:`mne.io.BaseRaw` object.

        Parameters
        ----------
        path:
            Absolute or relative path to the ``.snirf`` file.

        Raises
        ------
        SnirfLoadError
            When the file does not exist, is not a valid SNIRF, or MNE raises
            any exception during loading.
        """
        if not path.exists():
            raise SnirfLoadError(f"SNIRF file not found: {path}")

        try:
            raw: mne.io.BaseRaw = mne.io.read_raw_snirf(str(path), preload=True, verbose=False)
        except Exception as exc:  # noqa: BLE001
            raise SnirfLoadError(f"Failed to load SNIRF file '{path}': {exc}") from exc

        return raw
