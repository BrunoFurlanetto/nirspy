"""MNEAdapter -- thin wrapper around MNE-NIRS.

Provides stateless methods for I/O and signal-processing operations used by
concrete blocks.  Additional methods are added per-etapa.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mne
import mne.io
import mne.preprocessing.nirs
import numpy as np
from scipy.interpolate import UnivariateSpline

from nirspy.engine.exceptions import MNEOperationError, SnirfLoadError

if TYPE_CHECKING:
    from nirspy.blocks.analysis import ConditionWindow

logger = logging.getLogger(__name__)


def _label_segments(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label contiguous True segments in a boolean array.

    Returns
    -------
    labels:
        Array of same shape as *mask* where each contiguous True run
        is assigned a unique positive integer (1, 2, ...).  False
        entries are 0.
    n_segments:
        Total number of contiguous segments found.
    """
    labels = np.zeros_like(mask, dtype=int)
    seg_id = 0
    in_segment = False
    for i, val in enumerate(mask):
        if val and not in_segment:
            seg_id += 1
            in_segment = True
        elif not val:
            in_segment = False
        if val:
            labels[i] = seg_id
    return labels, seg_id


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
    # Motion Correction (v0.2)
    # ------------------------------------------------------------------

    def tddr(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply Temporal Derivative Distribution Repair (Fishburn et al., 2019).

        Parameters
        ----------
        raw:
            MNE Raw with fnirs_od channels.

        Returns
        -------
        mne.io.BaseRaw
            Motion-corrected Raw (same channel type).

        Raises
        ------
        MNEOperationError
            When MNE-NIRS raises any exception during TDDR.
        """
        try:
            return mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw)
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"tddr() failed: {exc}", mne_exception=exc
            ) from exc

    def spline_motion_correction(
        self,
        raw: mne.io.BaseRaw,
        threshold: float = 3.0,
        spline_order: int = 3,
    ) -> mne.io.BaseRaw:
        """Apply spline interpolation motion correction (Scholkmann et al., 2010).

        Custom implementation — MNE-NIRS does not provide this method.

        Algorithm
        ---------
        1. Compute the temporal derivative of each channel.
        2. Compute the z-score of the derivative.
        3. Identify artifact segments where |z| > *threshold*.
        4. For each artifact segment, fit a ``UnivariateSpline`` of order
           *spline_order* through the segment and subtract it from the
           original signal.

        Parameters
        ----------
        raw:
            MNE Raw with ``fnirs_od`` channels.
        threshold:
            Z-score cutoff for artifact detection (default 3.0).
        spline_order:
            Spline interpolation order, 1-5 (default 3 = cubic).

        Returns
        -------
        mne.io.BaseRaw
            Motion-corrected copy of the Raw object.

        Raises
        ------
        MNEOperationError
            When the correction fails for any reason.
        """
        try:
            corrected = raw.copy()
            data = corrected.get_data()  # (n_channels, n_times)
            sfreq = corrected.info["sfreq"]
            n_times = data.shape[1]

            if n_times < 4:
                # Too short for meaningful correction — return copy as-is
                return corrected

            times = np.arange(n_times) / sfreq

            for ch_idx in range(data.shape[0]):
                signal = data[ch_idx].copy()

                # Step 1: temporal derivative
                derivative = np.diff(signal)
                if len(derivative) == 0:
                    continue

                # Step 2: z-score of derivative
                d_std = np.std(derivative)
                if d_std == 0:
                    continue  # constant signal — nothing to correct
                d_mean = np.mean(derivative)
                z_scores = (derivative - d_mean) / d_std

                # Step 3: identify artifact samples
                artifact_mask = np.abs(z_scores) > threshold

                if not np.any(artifact_mask):
                    continue  # no artifacts detected

                # Step 4: find contiguous artifact segments and interpolate
                # Expand mask to original signal indices
                # derivative[i] corresponds to signal[i+1] - signal[i]
                # Mark both i and i+1 as artifact-affected
                signal_mask = np.zeros(n_times, dtype=bool)
                artifact_indices = np.where(artifact_mask)[0]
                signal_mask[artifact_indices] = True
                signal_mask[np.minimum(artifact_indices + 1, n_times - 1)] = True

                # Find contiguous segments
                labeled, n_segments = _label_segments(signal_mask)

                for seg_id in range(1, n_segments + 1):
                    seg_indices = np.where(labeled == seg_id)[0]
                    if len(seg_indices) < 2:
                        continue

                    start = max(0, seg_indices[0] - 1)
                    end = min(n_times - 1, seg_indices[-1] + 1)
                    seg_slice = slice(start, end + 1)
                    seg_times = times[seg_slice]
                    seg_signal = signal[seg_slice]

                    # Clamp spline order to available points - 1
                    k = min(spline_order, len(seg_times) - 1)
                    if k < 1:
                        continue

                    spline = UnivariateSpline(
                        seg_times, seg_signal, k=k, s=0
                    )
                    fitted = spline(seg_times)

                    # Subtract spline fit (artifact component) from signal
                    data[ch_idx, seg_slice] = signal[seg_slice] - fitted + np.mean(
                        seg_signal
                    )

            # SEC-INFO-03: validate shape before bypassing MNE setter
            assert data.shape == corrected.get_data().shape, (
                f"Shape mismatch after spline correction: "
                f"{data.shape} vs {corrected.get_data().shape}"
            )
            corrected._data = data
            return corrected
        except MNEOperationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"spline_motion_correction() failed: {exc}",
                mne_exception=exc,
            ) from exc


    def wavelet_motion_correction(
        self,
        raw: mne.io.BaseRaw,
        wavelet: str = "sym8",
        iqr_multiplier: float = 1.5,
    ) -> mne.io.BaseRaw:
        """Apply wavelet-based motion correction (Molavi & Dumont, 2012).

        Custom implementation using PyWavelets (pywt).

        Algorithm
        ---------
        1. For each channel: decompose via DWT (pywt.wavedec) to max level.
        2. For each detail-coefficient level: compute IQR.
        3. Soft-threshold: threshold = iqr_multiplier * IQR;
           coefs = sign(coefs) * max(|coefs| - threshold, 0).
        4. Reconstruct via pywt.waverec.

        Parameters
        ----------
        raw:
            MNE Raw with ``fnirs_od`` channels.
        wavelet:
            Wavelet family name (default "sym8").
        iqr_multiplier:
            IQR multiplier for soft-threshold (default 1.5).

        Returns
        -------
        mne.io.BaseRaw
            Motion-corrected copy of the Raw object.

        Raises
        ------
        MNEOperationError
            When the correction fails for any reason.
        """
        import pywt

        try:
            corrected = raw.copy()
            data = corrected.get_data()  # (n_channels, n_times)
            n_times = data.shape[1]

            if n_times < 4:
                # Too short for meaningful DWT -- return copy as-is
                return corrected

            for ch_idx in range(data.shape[0]):
                signal = data[ch_idx]

                # Step 1: DWT decomposition to max level
                coeffs = pywt.wavedec(signal, wavelet)

                # Step 2-3: soft-threshold each detail level (skip approx coeffs[0])
                for level_idx in range(1, len(coeffs)):
                    detail = coeffs[level_idx]
                    if len(detail) == 0:
                        continue

                    # IQR of this detail level
                    q75 = np.percentile(detail, 75)
                    q25 = np.percentile(detail, 25)
                    iqr = q75 - q25

                    if iqr == 0:
                        continue  # constant level -- no thresholding needed

                    threshold = iqr_multiplier * iqr

                    # Soft-threshold: sign(x) * max(|x| - threshold, 0)
                    coeffs[level_idx] = np.sign(detail) * np.maximum(
                        np.abs(detail) - threshold, 0.0
                    )

                # Step 4: reconstruct
                reconstructed = pywt.waverec(coeffs, wavelet)

                # pywt.waverec may return array 1 sample longer due to padding
                data[ch_idx] = reconstructed[:n_times]

            # SEC-INFO-03: validate shape before bypassing MNE setter
            assert data.shape == corrected.get_data().shape, (
                f"Shape mismatch after wavelet correction: "
                f"{data.shape} vs {corrected.get_data().shape}"
            )
            corrected._data = data
            return corrected
        except MNEOperationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"wavelet_motion_correction() failed: {exc}",
                mne_exception=exc,
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
        *,
        filter_bads: bool = True,
    ) -> dict[str, mne.Evoked]:
        """Average epochs per condition, returning dict of Evoked.

        Accepts a single ``mne.Epochs`` (legacy) or a
        ``dict[str, mne.Epochs]`` from create_epochs_per_condition.

        Parameters
        ----------
        epochs:
            Single Epochs or dict of per-condition Epochs.
        filter_bads:
            When True (default), channels listed in ``info['bads']``
            are dropped from each returned Evoked.  Set to False to
            preserve all channels (e.g. for QC dashboards).
        """
        try:
            if isinstance(epochs, dict):
                result = self._average_epochs_dict(epochs)
            else:
                result = self._average_epochs_single(epochs)
            if filter_bads:
                result = self._drop_bads_from_evoked(result)
            return result
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

    @staticmethod
    def _drop_bads_from_evoked(
        evoked_dict: dict[str, mne.Evoked],
    ) -> dict[str, mne.Evoked]:
        """Drop channels listed in info['bads'] from each Evoked.

        Returns a new dict; the original Evoked objects are copied so
        the caller retains access to the unfiltered data if needed.
        """
        filtered: dict[str, mne.Evoked] = {}
        for condition, evoked in evoked_dict.items():
            bads = evoked.info.get("bads", [])
            if bads:
                n_bads = len(bads)
                logger.debug(
                    "average_epochs filter_bads: dropping %d bad channel(s) "
                    "from condition %r: %s",
                    n_bads,
                    condition,
                    bads,
                )
                evoked_clean = evoked.copy()
                evoked_clean.pick(
                    picks="all", exclude="bads",
                )
                filtered[condition] = evoked_clean
            else:
                filtered[condition] = evoked
        return filtered
