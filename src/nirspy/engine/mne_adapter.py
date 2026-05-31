"""MNEAdapter -- thin wrapper around MNE-NIRS.

Provides stateless methods for I/O and signal-processing operations used by
concrete blocks.  Additional methods are added per-etapa.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mne
import mne.io
import mne.preprocessing.nirs
import numpy as np
from scipy.interpolate import UnivariateSpline

from nirspy.engine.exceptions import MNEOperationError, SnirfLoadError

if TYPE_CHECKING:
    import pandas as pd

    from nirspy.blocks.analysis import ConditionGroup, ConditionWindow

logger = logging.getLogger(__name__)


def _label_segments(
    mask: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], int]:
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


def _get_annotation_duration(
    raw: mne.io.BaseRaw,
    cond_name: str,
    onset_sample: int,
    sfreq: float,
) -> float:
    """Return the stimulus duration for a single event from raw annotations.

    Searches raw.annotations for a matching description + onset and returns
    the recorded duration. Falls back to 1.0 s when no match is found or
    when the stored duration is zero.

    Parameters
    ----------
    raw:
        MNE Raw object whose annotations are searched.
    cond_name:
        Condition name (annotation description) to match.
    onset_sample:
        Sample index of the event onset (relative to first_samp == 0).
    sfreq:
        Sampling frequency in Hz.

    Returns
    -------
    float
        Duration in seconds (> 0).
    """
    for ann in raw.annotations:
        if ann["description"] == cond_name:
            ann_onset_sample = int(ann["onset"] * sfreq)
            if abs(ann_onset_sample - onset_sample) < 2:  # tolerance 2 samples
                return float(ann["duration"]) if ann["duration"] > 0 else 1.0
    return 1.0


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
    # Signal Enhancement (v0.4)
    # ------------------------------------------------------------------

    def short_channel_regression(
        self,
        raw: mne.io.BaseRaw,
        max_dist: float = 0.015,
    ) -> mne.io.BaseRaw:
        """Regress out short-channel signals from long channels.

        Short-separation channels (source-detector distance <= max_dist)
        capture systemic physiology. This method uses linear regression to
        remove their contribution from long channels.

        Parameters
        ----------
        raw:
            MNE Raw with hbo/hbr channels (post Beer-Lambert).
        max_dist:
            Maximum source-detector distance in meters to classify a
            channel as 'short'. Default 0.015 m (15 mm).

        Returns
        -------
        mne.io.BaseRaw
            Raw with short-channel contributions regressed out of long
            channels. Short channels are dropped from the output.

        Raises
        ------
        MNEOperationError
            When MNE raises any exception during regression.
        """
        try:
            from mne_nirs.signal_enhancement import (
                short_channel_regression as _mne_nirs_scr,
            )

            return _mne_nirs_scr(raw, max_dist=max_dist)
        except ImportError:
            pass

        # Fallback: use mne.preprocessing.nirs if available (MNE >= 1.4)
        try:
            result = mne.preprocessing.nirs.short_channel_regression(
                raw, max_dist=max_dist
            )
            return result
        except AttributeError as exc:
            raise MNEOperationError(
                'short_channel_regression() requires mne-nirs or MNE >= 1.4. '
                'Neither mne_nirs.signal_enhancement.short_channel_regression '
                'nor mne.preprocessing.nirs.short_channel_regression is available.'
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f'short_channel_regression() failed: {exc}',
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

    @staticmethod
    def _events_by_indices(
        raw: mne.io.BaseRaw,
        indices: list[int],
    ) -> tuple[np.ndarray[Any, Any], dict[str, int]]:
        """Build an MNE events array from chronological annotation indices.

        Annotations are sorted by onset and assigned a chronological index
        (0 = first stim across *all* stim types). The returned events array
        contains only the rows corresponding to *indices*, preserving the
        original event codes so the group Epochs carries meaningful condition
        labels.

        Parameters
        ----------
        raw:
            MNE Raw object whose annotations define the events.
        indices:
            Chronological positions (0-based) into the sorted annotations.

        Returns
        -------
        events:
            MNE events array (n_selected, 3) sorted by onset sample.
        event_id:
            Mapping of condition_name -> event_code for the selected events.
        """
        all_events, auto_event_id = mne.events_from_annotations(
            raw, verbose=False
        )
        # Sorted list of (onset_s, description) from Raw annotations,
        # mirroring the order mne.events_from_annotations produces.
        sorted_annots = sorted(
            (ann["onset"], ann["description"])
            for ann in raw.annotations
        )
        # Resolve stim onset samples for each chronological index.
        sfreq = raw.info["sfreq"]
        first_sample = raw.first_samp

        selected_rows: list[np.ndarray[Any, Any]] = []
        selected_event_id: dict[str, int] = {}

        for idx in indices:
            if idx < 0 or idx >= len(sorted_annots):
                logger.warning(
                    "_events_by_indices: index %d out of range "
                    "(annotations length %d). Skipping.",
                    idx,
                    len(sorted_annots),
                )
                continue
            onset_s, description = sorted_annots[idx]
            # Find matching row in all_events by sample proximity.
            onset_sample = int(round(onset_s * sfreq)) + first_sample
            # Tolerance: ±2 samples to handle floating-point rounding.
            tol = 2
            code = auto_event_id.get(description)
            if code is None:
                logger.warning(
                    "_events_by_indices: description %r not found in "
                    "event_id. Skipping index %d.",
                    description,
                    idx,
                )
                continue
            mask = (
                (np.abs(all_events[:, 0] - onset_sample) <= tol)
                & (all_events[:, 2] == code)
            )
            matching = all_events[mask]
            if len(matching) == 0:
                logger.warning(
                    "_events_by_indices: no MNE event found near sample "
                    "%d (onset %.3fs, code %d). Skipping index %d.",
                    onset_sample,
                    onset_s,
                    code,
                    idx,
                )
                continue
            selected_rows.append(matching[0])
            selected_event_id[description] = code

        if not selected_rows:
            return np.zeros((0, 3), dtype=int), {}

        selected_events = np.vstack(selected_rows)
        # Sort by sample number (MNE convention)
        order = np.argsort(selected_events[:, 0])
        return selected_events[order], selected_event_id

    def create_epochs_per_group(
        self,
        raw: mne.io.BaseRaw,
        groups: dict[str, ConditionGroup],
        *,
        reject: dict[str, float] | None,
    ) -> dict[str, mne.Epochs]:
        """Create one Epochs per condition group.

        Each group aggregates multiple SNIRF condition keys under a
        single label. The resulting dict is keyed by group label.

        Supports two modes per group (D8):
        - **condition_names mode**: all occurrences of the listed SNIRF
          condition keys are included (T-024 behaviour).
        - **event_indices mode**: only specific occurrences identified by
          their chronological index in ``raw.annotations`` are included
          (T-030 timeline selection).

        Parameters
        ----------
        raw:
            MNE Raw with annotations marking stimulus events.
        groups:
            Mapping of group label to :class:`ConditionGroup`.
        reject:
            Channel-type rejection thresholds.

        Returns
        -------
        dict[str, mne.Epochs]
            One Epochs per group label.
        """
        try:
            events, auto_event_id = mne.events_from_annotations(
                raw, verbose=False
            )
            result: dict[str, mne.Epochs] = {}
            for label, group in groups.items():
                # --- event_indices mode (T-030) ---
                if group.event_indices:
                    group_events, subset_event_id = self._events_by_indices(
                        raw, group.event_indices
                    )
                    if len(group_events) == 0:
                        logger.warning(
                            "create_epochs_per_group: group %r produced no "
                            "events from event_indices %s. Skipping.",
                            label,
                            group.event_indices,
                        )
                        continue
                # --- condition_names mode (T-024) ---
                else:
                    subset_event_id = {
                        cond: auto_event_id[cond]
                        for cond in group.condition_names
                        if cond in auto_event_id
                    }
                    if not subset_event_id:
                        logger.warning(
                            "create_epochs_per_group: group %r has no matching "
                            "conditions in event_id. Skipping.",
                            label,
                        )
                        continue
                    # Filter events to only include this group
                    valid_codes = set(subset_event_id.values())
                    mask = np.isin(events[:, 2], list(valid_codes))
                    group_events = events[mask]
                    if len(group_events) == 0:
                        continue

                epochs = mne.Epochs(
                    raw,
                    group_events,
                    event_id=subset_event_id,
                    tmin=group.tmin,
                    tmax=group.tmax,
                    baseline=(group.baseline_tmin, group.baseline_tmax),
                    reject=reject,
                    preload=True,
                    verbose=False,
                )
                result[label] = epochs
            return result
        except MNEOperationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"create_epochs_per_group() failed: {exc}",
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

    # ------------------------------------------------------------------
    # Export (v0.4)
    # ------------------------------------------------------------------

    def evoked_to_dataframe(
        self,
        evoked_dict: dict[str, mne.Evoked],
    ) -> pd.DataFrame:
        """Convert a dict of Evoked objects to a single DataFrame.

        Each condition produces rows with columns:
        - time (seconds)
        - channel name
        - value (concentration in mol/L)
        - condition (label)
        - ch_type (hbo/hbr)

        Parameters
        ----------
        evoked_dict:
            Mapping of condition name to mne.Evoked.

        Returns
        -------
        pd.DataFrame
            Long-format table suitable for statistical analysis.

        Raises
        ------
        MNEOperationError
            When conversion fails.
        """
        import pandas as pd

        try:
            frames: list[pd.DataFrame] = []
            for condition, evoked in evoked_dict.items():
                df = evoked.to_data_frame(time_format=None)
                # MNE returns wide format: columns = channel names, index = time
                # Melt to long format
                df = df.reset_index()
                df_long = df.melt(
                    id_vars=["time"],
                    var_name="channel",
                    value_name="value",
                )
                df_long["condition"] = condition
                # Add channel type info
                ch_type_map = dict(
                    zip(evoked.ch_names, evoked.get_channel_types(), strict=False)
                )
                df_long["ch_type"] = df_long["channel"].map(ch_type_map)
                frames.append(df_long)

            if not frames:
                raise MNEOperationError(
                    "evoked_to_dataframe(): empty evoked_dict, nothing to convert."
                )

            result = pd.concat(frames, ignore_index=True)
            return result
        except MNEOperationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"evoked_to_dataframe() failed: {exc}",
                mne_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # GLM Analysis (v0.4 - T-034)
    # ------------------------------------------------------------------

    def run_glm(
        self,
        raw: mne.io.BaseRaw,
        *,
        event_id: dict[str, int] | None = None,
        drift_model: str = "cosine",
        high_pass: float = 0.01,
        hrf_model: str = "glover",
        noise_model: str = "ar1",
        condition_durations: dict[str, float] | None = None,
        per_condition_groups: dict[str, list[str]] | None = None,
    ) -> Any:
        """Run first-level GLM on haemodynamic Raw data.

        Builds a design matrix from stimulus annotations using nilearn,
        then fits the GLM via mne_nirs.statistics.run_glm.

        Parameters
        ----------
        raw:
            MNE Raw with hbo/hbr channels and stimulus annotations.
        event_id:
            Optional mapping of condition names to event codes.
            If None, all annotations are used.
        drift_model:
            Drift model for design matrix ('cosine' or 'polynomial').
        high_pass:
            High-pass cutoff for cosine drift (Hz). Default 0.01.
        hrf_model:
            HRF model ('glover', 'spm', 'fir', 'glover + derivative', etc.).
        noise_model:
            Noise model for GLM ('ar1' or 'ols').
        condition_durations:
            Optional per-condition stimulus duration in seconds.
            When None, duration is read from raw annotations; falls back to 1.0 s.
        per_condition_groups:
            Optional grouping of conditions for the design matrix.
            Maps group label -> list of condition names to merge under that label.
            When None, each condition is modelled independently.

        Returns
        -------
        GLMResult
            Domain-layer container with coefficients, t-stats, p-values.

        Raises
        ------
        MNEOperationError
            When GLM fitting fails for any reason.
        """
        from nirspy.domain.glm_result import GLMResult

        try:
            from mne_nirs.statistics import run_glm as _mne_nirs_run_glm
            from nilearn.glm.first_level import make_first_level_design_matrix

            # Build frame times from raw duration and sampling rate
            sfreq = raw.info["sfreq"]
            n_times = raw.n_times
            frame_times = np.arange(n_times) / sfreq

            # Extract events as nilearn-compatible DataFrame
            events_array, auto_event_id = mne.events_from_annotations(
                raw, verbose=False
            )
            used_event_id = event_id if event_id is not None else auto_event_id

            if len(events_array) == 0:
                raise MNEOperationError(
                    "run_glm(): no events found in raw annotations."
                )

            # Build nilearn events DataFrame
            import pandas as pd

            event_rows: list[dict[str, Any]] = []
            # Invert event_id to get code -> name mapping
            code_to_name = {v: k for k, v in used_event_id.items()}
            first_samp = raw.first_samp

            for event in events_array:
                code = int(event[2])
                if code in code_to_name:
                    onset_sample = int(event[0]) - first_samp
                    cond_name = code_to_name[code]
                    if condition_durations and cond_name in condition_durations:
                        duration = condition_durations[cond_name]
                    else:
                        duration = _get_annotation_duration(
                            raw, cond_name, onset_sample, sfreq
                        )
                    event_rows.append({
                        "onset": onset_sample / sfreq,
                        "duration": duration,
                        "trial_type": cond_name,
                    })

            if not event_rows:
                raise MNEOperationError(
                    "run_glm(): no matching events found for the given event_id."
                )

            events_df = pd.DataFrame(event_rows)

            # Apply condition grouping: remap trial_type before building design matrix
            if per_condition_groups:
                cond_to_group: dict[str, str] = {}
                for group_label, cond_names in per_condition_groups.items():
                    for cn in cond_names:
                        cond_to_group[cn] = group_label
                events_df["trial_type"] = events_df["trial_type"].map(
                    lambda x: cond_to_group.get(x, x)
                )

            # Build design matrix
            # oversampling=1: nilearn default (50) creates n_times*50 intermediate
            # array — extremely slow for fNIRS at 10Hz. At 10Hz the frame_times
            # already provide sufficient HRF resolution without oversampling.
            design_matrix = make_first_level_design_matrix(
                frame_times=frame_times,
                events=events_df,
                hrf_model=hrf_model,
                drift_model=drift_model,
                high_pass=high_pass,
                oversampling=1,
            )

            # Run GLM via mne-nirs
            glm_est = _mne_nirs_run_glm(
                raw,
                design_matrix,
                noise_model=noise_model,
            )

            # Extract results from RegressionResults
            # glm_est is a RegressionResults object
            df = glm_est.to_dataframe()
            ch_names = glm_est.ch_names
            regressor_names = list(design_matrix.columns)

            # Build matrices from the dataframe
            n_regressors = len(regressor_names)
            n_channels = len(ch_names)

            theta = np.zeros((n_regressors, n_channels))
            t_stats_mat = np.zeros((n_regressors, n_channels))
            p_values_mat = np.zeros((n_regressors, n_channels))

            # The to_dataframe() returns columns like:
            # ch_name, Condition, theta, t_stat, ...
            for i, reg in enumerate(regressor_names):
                reg_df = df[df["Condition"] == reg]
                for j, ch in enumerate(ch_names):
                    ch_row = reg_df[reg_df["ch_name"] == ch]
                    if len(ch_row) > 0:
                        theta[i, j] = float(ch_row["theta"].iloc[0])
                        for t_col in ("t", "t_stat"):
                            if t_col in ch_row.columns:
                                t_stats_mat[i, j] = float(ch_row[t_col].iloc[0])
                                break
                        if "p_value" in ch_row.columns:
                            p_values_mat[i, j] = float(
                                ch_row["p_value"].iloc[0]
                            )

            # MSE per channel
            mse = np.array(glm_est.MSE())

            return GLMResult(
                theta=theta,
                t_stats=t_stats_mat,
                p_values=p_values_mat,
                mse=mse,
                channel_names=list(ch_names),
                regressor_names=regressor_names,
                design_matrix=design_matrix.values,
                noise_model=noise_model,
                metadata={
                    "drift_model": drift_model,
                    "high_pass": high_pass,
                    "hrf_model": hrf_model,
                    "n_events": len(event_rows),
                    "conditions": list(events_df["trial_type"].unique()),
                },
            )
        except MNEOperationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MNEOperationError(
                f"run_glm() failed: {exc}", mne_exception=exc
            ) from exc
