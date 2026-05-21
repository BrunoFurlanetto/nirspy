"""MNEAdapter -- thin wrapper around MNE-NIRS.

Provides stateless methods for I/O and signal-processing operations used by
concrete blocks.  Additional methods are added per-etapa.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mne
import mne.io
import mne.preprocessing.nirs

from nirspy.engine.exceptions import MNEOperationError, SnirfLoadError

if TYPE_CHECKING:
    from nirspy.blocks.analysis import ConditionWindow


class RawWrapper:
    """Lightweight container that pairs a mne.io.BaseRaw with its source path."""

    def __init__(self, raw: mne.io.BaseRaw, source_path: Path) -> None:
        self.raw = raw
        self.source_path = source_path

    def __repr__(self) -> str:
        return f"RawWrapper(path={self.source_path!r}, n_channels={len(self.raw.ch_names)})"


class MNEAdapter:
    """Facade over MNE / MNE-NIRS I/O and signal-processing operations.

    All methods are stateless -- the class holds no mutable state and can be
    instantiated once and reused across pipeline runs.
    """

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_snirf(self, path: Path) -> mne.io.BaseRaw:
        """Load a SNIRF file and return a mne.io.BaseRaw object.

        Raises
        ------
        SnirfLoadError
            When the file does not exist, is not a valid SNIRF, or MNE raises
            any exception during loading.
        """
        if not path.exists():
            raise SnirfLoadError(f"SNIRF file not found: {path}")

        try:
            raw: mne.io.BaseRaw = mne.io.read_raw_snirf(
                str(path), preload=True, verbose=False
            )
        except Exception as exc:  # noqa: BLE001
            raise SnirfLoadError(
                f"Failed to load SNIRF file '{path}': {exc}", mne_exception=exc
            ) from exc

        return raw

    # ------------------------------------------------------------------
    # Preprocessing (Etapa 2)
    # ------------------------------------------------------------------

    def raw_to_od(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Convert raw intensity to optical density via log-ratio.

        Parameters
        ----------
        raw:
            MNE Raw with ``fnirs_cw_amplitude`` channels.

        Returns
        -------
        mne.io.BaseRaw
            Raw with ``fnirs_od`` channel type.

        Raises
        ------
        MNEOperationError
            When MNE raises any exception during conversion.
        """
        try:
            return mne.preprocessing.nirs.optical_density(raw)
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"optical_density() failed: {exc}", mne_exception=exc
            ) from exc

    def beer_lambert(self, raw: mne.io.BaseRaw, ppf: float = 6.0) -> mne.io.BaseRaw:
        """Apply modified Beer-Lambert Law to convert OD to HbO/HbR.

        Parameters
        ----------
        raw:
            MNE Raw with ``fnirs_od`` channels.
        ppf:
            Partial Pathlength Factor. Default 6.0 per Yucel et al. 2021.

        Returns
        -------
        mne.io.BaseRaw
            Raw with ``hbo`` and ``hbr`` channel types.

        Raises
        ------
        MNEOperationError
            When MNE raises any exception during conversion.
        """
        try:
            return mne.preprocessing.nirs.beer_lambert_law(raw, ppf=ppf)
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"beer_lambert_law() failed: {exc}", mne_exception=exc
            ) from exc

    def bandpass_filter(
        self,
        raw: mne.io.BaseRaw,
        l_freq: float | None = 0.01,
        h_freq: float | None = 0.5,
        method: str = "iir",
        iir_params: dict[str, object] | None = None,
    ) -> mne.io.BaseRaw:
        """Apply bandpass filter to Raw data.

        Parameters
        ----------
        raw:
            MNE Raw object (any fNIRS channel type).
        l_freq:
            Low cutoff frequency in Hz. None disables highpass.
        h_freq:
            High cutoff frequency in Hz. None disables lowpass.
        method:
            "iir" or "fir".
        iir_params:
            Optional dict passed directly to raw.filter(iir_params=...).

        Returns
        -------
        mne.io.BaseRaw
            Filtered copy of the Raw object.

        Raises
        ------
        MNEOperationError
            When MNE raises any exception during filtering.
        """
        try:
            raw_filtered = raw.copy()
            raw_filtered.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                method=method,
                iir_params=iir_params,
                verbose=False,
            )
            return raw_filtered
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"bandpass_filter() failed: {exc}", mne_exception=exc
            ) from exc

    # ------------------------------------------------------------------
    # Quality Control (Etapa 3)
    # ------------------------------------------------------------------

    def scalp_coupling_index(self, raw: mne.io.BaseRaw) -> dict[str, float]:
        """Compute Scalp Coupling Index per channel (Pollonini et al., 2014).

        Parameters
        ----------
        raw:
            MNE Raw with ``fnirs_od`` channels.

        Returns
        -------
        dict[str, float]
            Mapping of channel name to SCI value in [0, 1].

        Raises
        ------
        MNEOperationError
            When MNE raises any exception during computation.
        """
        try:
            sci_array = mne.preprocessing.nirs.scalp_coupling_index(raw)
            return dict(zip(raw.ch_names, sci_array.tolist(), strict=False))
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"scalp_coupling_index() failed: {exc}", mne_exception=exc
            ) from exc

    # ------------------------------------------------------------------
    # Analysis (Etapa 3)
    # ------------------------------------------------------------------

    def create_epochs(
        self,
        raw: mne.io.BaseRaw,
        tmin: float = -2.0,
        tmax: float = 18.0,
        baseline_tmin: float = -2.0,
        baseline_tmax: float = 0.0,
        reject: dict[str, float] | None = None,
        event_id: dict[str, int] | None = None,
    ) -> mne.Epochs:
        """Create epochs from annotations in the Raw object.

        Parameters
        ----------
        raw:
            MNE Raw with annotations marking stimulus events.
        tmin, tmax:
            Epoch window relative to event onset (seconds).
        baseline_tmin, baseline_tmax:
            Baseline correction window.
        reject:
            Channel-type rejection thresholds (e.g. {"hbo": 80e-6}).
        event_id:
            Optional mapping of condition names to event codes.
            If None, all annotation-derived events are used.

        Returns
        -------
        mne.Epochs
            Epoched data.

        Raises
        ------
        MNEOperationError
            When MNE raises any exception during epoching.
        """
        try:
            events, auto_event_id = mne.events_from_annotations(
                raw, verbose=False
            )
            used_event_id = event_id if event_id is not None else auto_event_id
            epochs = mne.Epochs(
                raw,
                events,
                event_id=used_event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=(baseline_tmin, baseline_tmax),
                reject=reject,
                preload=True,
                verbose=False,
            )
            return epochs
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"create_epochs() failed: {exc}", mne_exception=exc
            ) from exc


    def create_epochs_per_condition(
        self,
        raw: mne.io.BaseRaw,
        event_id: dict[str, int],
        *,
        default_window: tuple[float, float, float, float],
        per_condition_windows: dict[str, ConditionWindow],
        reject: dict[str, float] | None,
    ) -> dict[str, mne.Epochs]:
        """Create one Epochs per condition with per-condition windows.

        MNE does not support mixed temporal windows in a single Epochs.
        This method loops over conditions, creating a separate Epochs
        for each using its override or the default_window fallback.
        """
        try:
            events, _auto_event_id = mne.events_from_annotations(
                raw, verbose=False
            )
            result: dict[str, mne.Epochs] = {}
            for cond, code in event_id.items():
                mask = events[:, 2] == code
                cond_events = events[mask]
                if len(cond_events) == 0:
                    continue
                if cond in per_condition_windows:
                    w = per_condition_windows[cond]
                    tmin, tmax = w.tmin, w.tmax
                    bl_tmin, bl_tmax = w.baseline_tmin, w.baseline_tmax
                else:
                    tmin, tmax, bl_tmin, bl_tmax = default_window
                epochs = mne.Epochs(
                    raw,
                    cond_events,
                    event_id={cond: code},
                    tmin=tmin,
                    tmax=tmax,
                    baseline=(bl_tmin, bl_tmax),
                    reject=reject,
                    preload=True,
                    verbose=False,
                )
                result[cond] = epochs
            return result
        except MNEOperationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"create_epochs_per_condition() failed: {exc}",
                mne_exception=exc,
            ) from exc

    def average_epochs(
        self,
        epochs: mne.Epochs | dict[str, mne.Epochs],
    ) -> dict[str, mne.Evoked]:
        """Average epochs per condition, returning dict of Evoked.

        Accepts a single ``mne.Epochs`` (legacy) or a
        ``dict[str, mne.Epochs]`` from create_epochs_per_condition.
        """
        try:
            if isinstance(epochs, dict):
                return self._average_epochs_dict(epochs)
            return self._average_epochs_single(epochs)
        except MNEOperationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"average_epochs() failed: {exc}", mne_exception=exc
            ) from exc

    def _average_epochs_single(
        self, epochs: mne.Epochs
    ) -> dict[str, mne.Evoked]:
        """Average a single Epochs object per condition."""
        result: dict[str, mne.Evoked] = {}
        for condition in epochs.event_id:
            condition_epochs = epochs[condition]
            if len(condition_epochs) == 0:
                continue
            evoked = condition_epochs.average()
            result[condition] = evoked
        if not result:
            raise MNEOperationError(
                "average_epochs() produced no evoked: all conditions "
                "had every epoch rejected. Loosen reject_by_amplitude "
                "or raise amplitude_threshold."
            )
        return result

    def _average_epochs_dict(
        self, epochs_dict: dict[str, mne.Epochs]
    ) -> dict[str, mne.Evoked]:
        """Average a dict of per-condition Epochs objects."""
        result: dict[str, mne.Evoked] = {}
        for condition, cond_epochs in epochs_dict.items():
            if len(cond_epochs) == 0:
                continue
            evoked = cond_epochs.average()
            result[condition] = evoked
        if not result:
            raise MNEOperationError(
                "average_epochs() produced no evoked: all conditions "
                "had every epoch rejected. Loosen reject_by_amplitude "
                "or raise amplitude_threshold."
            )
        return result
