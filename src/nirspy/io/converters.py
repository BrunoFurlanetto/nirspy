"""Bidirectional converter between .nirs (HOMER2/3 MAT-file) and .snirf (SNIRF 1.1 HDF5).

SNIRF specification: https://github.com/fNIRS/snirf/blob/master/snirf_specification.md

Design:
    - ``NirsData`` is a pivot dataclass read from either format and written to the other.
    - Public API: ``nirs_to_snirf`` and ``snirf_to_nirs``.
    - No pysnirf2 / snirf — both are GPL-3 (ADR-004). Uses h5py (BSD-3) directly.
    - No MNE, Dash or Plotly imports (ADR-005).
    - Fields with no equivalent in the target format emit ``UserWarning`` and are dropped.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io

from nirspy.domain.exceptions import (
    ConverterError,
    NirsDataError,
    NirsParseError,
    NirsWriteError,
    SnirfParseError,
    SnirfWriteError,
)

__all__ = [
    "NirsData",
    "MeasurementChannel",
    "StimEvent",
    "nirs_to_snirf",
    "snirf_to_nirs",
]


# ---------------------------------------------------------------------------
# Security constants (S-001)
# ---------------------------------------------------------------------------

#: Maximum number of samples (time points) allowed in a single dataset.
MAX_SAMPLES: int = 100_000_000  # 100M

#: Maximum number of channels allowed in a single dataset.
MAX_CHANNELS: int = 100_000  # 100K

#: Maximum number of dimensions allowed for data arrays.
MAX_NDIM: int = 4

# ---------------------------------------------------------------------------
# PII fields that can be stripped (I-001)
# ---------------------------------------------------------------------------

_PII_FIELDS: set[str] = {
    "SubjectID",
    "DateOfBirth",
    "SubjectName",
    "PatientName",
    "PatientID",
}

# ---------------------------------------------------------------------------
# Pivot dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasurementChannel:
    """Source-detector-wavelength measurement channel mapping.

    All indices are 1-based to match the SNIRF 1.1 / HOMER convention.
    ``data_type`` is always 1 (raw CW amplitude) because .nirs files produced
    by HOMER2/3 exclusively contain continuous-wave raw data.
    """

    source_index: int
    detector_index: int
    wavelength_index: int  # 1-based index into NirsData.wavelengths
    data_type: int = 1  # SNIRF dataType: 1 = raw CW amplitude (D5: hard-coded)


@dataclass(frozen=True)
class StimEvent:
    """Single stimulus event."""

    name: str
    onset: float  # seconds from session start
    duration: float  # seconds
    value: float = 1.0


@dataclass
class NirsData:
    """Canonical pivot representation.  Lives only in memory during conversion.

    Attributes:
        data_matrix:   (n_samples, n_channels) float64 array.
        time_vector:   (n_samples,) float64 array in seconds (D6: discrete vector).
        wavelengths:   List of wavelength values in nm, e.g. [760.0, 850.0].
        source_pos:    (n_sources, 3) float64 array in mm.
        detector_pos:  (n_detectors, 3) float64 array in mm.
        meas_list:     One MeasurementChannel per data column.
        stim_events:   Stimulus events; empty list when no stimuli.
        metadata:      Free-form metadata: SubjectID, SessionID, etc.
    """

    data_matrix: np.ndarray
    time_vector: np.ndarray
    wavelengths: list[float]
    source_pos: np.ndarray
    detector_pos: np.ndarray
    meas_list: list[MeasurementChannel]
    stim_events: list[StimEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Check invariants.  Raises ``NirsDataError`` if any is violated."""
        n_samples, n_channels = self.data_matrix.shape
        if self.time_vector.shape != (n_samples,):
            raise NirsDataError(
                f"time_vector length {self.time_vector.shape[0]} does not match "
                f"data_matrix n_samples {n_samples}."
            )
        if n_channels != len(self.meas_list):
            raise NirsDataError(
                f"meas_list length {len(self.meas_list)} does not match "
                f"data_matrix n_channels {n_channels}."
            )
        if self.source_pos.ndim != 2 or self.source_pos.shape[1] != 3:
            raise NirsDataError(
                f"source_pos must have shape (n_sources, 3), got {self.source_pos.shape}."
            )
        if self.detector_pos.ndim != 2 or self.detector_pos.shape[1] != 3:
            raise NirsDataError(
                f"detector_pos must have shape (n_detectors, 3), got {self.detector_pos.shape}."
            )
        if not self.wavelengths:
            raise NirsDataError("wavelengths list must not be empty.")
        for i, ch in enumerate(self.meas_list):
            if ch.wavelength_index < 1 or ch.wavelength_index > len(self.wavelengths):
                raise NirsDataError(
                    f"meas_list[{i}].wavelength_index {ch.wavelength_index} out of range "
                    f"[1, {len(self.wavelengths)}]."
                )
        if np.any(np.isnan(self.data_matrix)):
            warnings.warn(
                "data_matrix contains NaN values. They will be preserved as-is in the output.",
                UserWarning,
                stacklevel=3,
            )



# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------


def _validate_array_shape(
    shape: tuple[int, ...],
    context: str,
    path: Path,
) -> None:
    """Validate array shape against security limits (S-001).

    Raises ConverterError if shape exceeds MAX_SAMPLES, MAX_CHANNELS, or MAX_NDIM.
    """
    if len(shape) > MAX_NDIM:
        raise ConverterError(
            f"Array '{context}' has {len(shape)} dimensions (max {MAX_NDIM}): {path}"
        )
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    if total_elements > MAX_SAMPLES:
        raise ConverterError(
            f"Array '{context}' has {total_elements:,} elements "
            f"(max {MAX_SAMPLES:,}): {path}"
        )
    if len(shape) >= 2 and shape[1] > MAX_CHANNELS:
        raise ConverterError(
            f"Array '{context}' has {shape[1]:,} channels "
            f"(max {MAX_CHANNELS:,}): {path}"
        )


def _check_external_links(f: h5py.File, path: Path) -> None:
    """Reject HDF5 files containing ExternalLinks (S-002).

    Iterates all items recursively and raises ConverterError if any
    h5py.ExternalLink is found.
    """

    def _check_group(group: h5py.Group) -> None:
        for key in group:
            link = group.get(key, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                raise ConverterError(
                    f"Security: external HDF5 link detected at '{group.name}/{key}' "
                    f"in '{path}'. External links are not allowed (S-002)."
                )
            item = group.get(key)
            if isinstance(item, h5py.Group):
                _check_group(item)

    _check_group(f)


def _atomic_create_output(path: Path, overwrite: bool) -> None:
    """Atomically verify output path does not exist (S-003).

    Uses os.open with O_CREAT|O_EXCL to prevent TOCTOU race conditions.
    When overwrite=True, skips the atomic check.
    """
    if overwrite:
        if path.exists():
            path.unlink()
        return

    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        # Remove the empty lock file - actual write happens after
        path.unlink()
    except FileExistsError:
        raise ConverterError(
            f"Output file already exists (use overwrite=True to replace): {path}"
        ) from None


def _strip_pii_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove PII fields from metadata dict (I-001).

    Returns a new dict with PII fields removed.
    """
    return {k: v for k, v in metadata.items() if k not in _PII_FIELDS}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nirs_to_snirf(
    input_path: Path | str,
    output_path: Path | str,
    overwrite: bool = False,
    *,
    strip_pii: bool = False,
) -> None:
    """Convert a .nirs (HOMER2/3 MAT-file) to .snirf (SNIRF 1.1 HDF5).

    Args:
        input_path:  Path to the .nirs input file.
        output_path: Path for the .snirf output file.
        overwrite:   If ``False`` (default), raises ``ConverterError`` when
                     ``output_path`` already exists.

    Raises:
        ConverterError:   ``input_path`` does not exist, wrong extension, or
                          ``output_path`` exists with ``overwrite=False``.
        NirsParseError:   .nirs file cannot be read or has invalid structure.
        NirsDataError:    Pivot representation violates invariants.
        SnirfWriteError:  Writing the .snirf file failed.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    _validate_input(in_path, ".nirs")
    _atomic_create_output(out_path, overwrite)

    data = _parse_nirs(in_path)
    if strip_pii:
        data.metadata = _strip_pii_from_metadata(data.metadata)
    data.validate()
    _write_snirf(data, out_path)


def snirf_to_nirs(
    input_path: Path | str,
    output_path: Path | str,
    overwrite: bool = False,
    *,
    strip_pii: bool = False,
) -> None:
    """Convert a .snirf (SNIRF 1.1 HDF5) to .nirs (HOMER2/3 MAT-file v5).

    Fields present in SNIRF but without an equivalent in .nirs are discarded
    with a ``UserWarning`` identifying each dropped field.

    Args:
        input_path:  Path to the .snirf input file.
        output_path: Path for the .nirs output file.
        overwrite:   If ``False`` (default), raises ``ConverterError`` when
                     ``output_path`` already exists.

    Raises:
        ConverterError:  ``input_path`` does not exist, wrong extension, or
                         ``output_path`` exists with ``overwrite=False``.
        SnirfParseError: .snirf file is invalid or not SNIRF 1.1.
        NirsDataError:   Pivot representation violates invariants.
        NirsWriteError:  Writing the .nirs file failed.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    _validate_input(in_path, ".snirf")
    _atomic_create_output(out_path, overwrite)

    data = _parse_snirf(in_path)
    if strip_pii:
        data.metadata = _strip_pii_from_metadata(data.metadata)
    data.validate()
    _write_nirs(data, out_path)


# ---------------------------------------------------------------------------
# Internal helpers — validation
# ---------------------------------------------------------------------------


def _validate_input(path: Path, expected_suffix: str) -> None:
    """Raise ``ConverterError`` for missing file or wrong extension."""
    if not path.exists():
        raise ConverterError(f"Input file not found: {path}")
    if path.suffix.lower() != expected_suffix:
        raise ConverterError(
            f"Expected a {expected_suffix} file, got '{path.suffix}': {path}"
        )


def _validate_output(path: Path, overwrite: bool) -> None:
    """Raise ``ConverterError`` if output exists and overwrite is False."""
    if path.exists() and not overwrite:
        raise ConverterError(
            f"Output file already exists (use overwrite=True to replace): {path}"
        )


# ---------------------------------------------------------------------------
# Internal helpers — parsing
# ---------------------------------------------------------------------------


def _parse_nirs(path: Path) -> NirsData:
    """Read a .nirs MAT-file and return a ``NirsData`` pivot object.

    Supports MAT v5 and v7 (scipy.io.loadmat handles both).
    HOMER2/3 format expected; MAT v4 is not supported.
    """
    try:
        mat: dict[str, Any] = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        raise NirsParseError(f"Cannot read .nirs file: {exc}", path=path) from exc

    # --- data matrix -------------------------------------------------------
    if "d" not in mat:
        raise NirsParseError("Missing required field 'd' (data matrix).", path=path)
    # S-001: Check shape before allocating full array
    d_raw = mat["d"]
    if hasattr(d_raw, "shape"):
        _validate_array_shape(d_raw.shape, "d (data matrix)", path)

    data_matrix = np.array(d_raw, dtype=np.float64)
    if data_matrix.ndim == 1:
        data_matrix = data_matrix.reshape(-1, 1)

    # --- time vector -------------------------------------------------------
    if "t" not in mat:
        raise NirsParseError("Missing required field 't' (time vector).", path=path)
    time_vector = np.array(mat["t"], dtype=np.float64).ravel()

    # --- SD structure ------------------------------------------------------
    if "SD" not in mat:
        raise NirsParseError("Missing required field 'SD' (source-detector structure).", path=path)
    sd = mat["SD"]

    # wavelengths
    try:
        wavelengths = [float(w) for w in np.atleast_1d(sd.Lambda).ravel()]
    except AttributeError as exc:
        raise NirsParseError("SD.Lambda not found in .nirs file.", path=path) from exc

    # source positions
    try:
        src_pos_raw = np.array(sd.SrcPos, dtype=np.float64)
        if src_pos_raw.ndim == 1:
            src_pos_raw = src_pos_raw.reshape(1, -1)
    except AttributeError as exc:
        raise NirsParseError("SD.SrcPos not found in .nirs file.", path=path) from exc

    if src_pos_raw.shape[1] == 2:
        warnings.warn(
            f"SD.SrcPos in '{path.name}' has 2D coordinates. "
            "Elevating to 3D by appending z=0.0. Spatial visualisations may be affected.",
            UserWarning,
            stacklevel=3,
        )
        src_pos_raw = np.hstack([src_pos_raw, np.zeros((src_pos_raw.shape[0], 1))])
    source_pos = src_pos_raw

    # detector positions
    try:
        det_pos_raw = np.array(sd.DetPos, dtype=np.float64)
        if det_pos_raw.ndim == 1:
            det_pos_raw = det_pos_raw.reshape(1, -1)
    except AttributeError as exc:
        raise NirsParseError("SD.DetPos not found in .nirs file.", path=path) from exc

    if det_pos_raw.shape[1] == 2:
        warnings.warn(
            f"SD.DetPos in '{path.name}' has 2D coordinates. "
            "Elevating to 3D by appending z=0.0. Spatial visualisations may be affected.",
            UserWarning,
            stacklevel=3,
        )
        det_pos_raw = np.hstack([det_pos_raw, np.zeros((det_pos_raw.shape[0], 1))])
    detector_pos = det_pos_raw

    # measurement list (SD.MeasList: n_channels x 4 — src, det, ?, wl_idx)
    try:
        meas_list_raw = np.array(sd.MeasList, dtype=int)
        if meas_list_raw.ndim == 1:
            meas_list_raw = meas_list_raw.reshape(1, -1)
    except AttributeError as exc:
        raise NirsParseError("SD.MeasList not found in .nirs file.", path=path) from exc

    meas_list = [
        MeasurementChannel(
            source_index=int(row[0]),
            detector_index=int(row[1]),
            wavelength_index=int(row[3]),
        )
        for row in meas_list_raw
    ]

    # --- stimulus matrix ---------------------------------------------------
    stim_events: list[StimEvent] = []
    if "s" in mat:
        s_raw = np.array(mat["s"], dtype=np.float64)
        if s_raw.ndim == 1:
            s_raw = s_raw.reshape(-1, 1)
        for col_idx in range(s_raw.shape[1]):
            col = s_raw[:, col_idx]
            nonzero_idx = np.where(col != 0)[0]
            if nonzero_idx.size == 0:
                continue
            cond_name = str(float(col_idx + 1))
            # try to read condition names from CondNames field if present
            try:
                cond_names_field = sd.CondNames
                names_arr = np.atleast_1d(cond_names_field).ravel()
                if col_idx < len(names_arr):
                    cond_name = str(names_arr[col_idx])
            except AttributeError:
                pass
            dt = float(time_vector[1] - time_vector[0]) if len(time_vector) > 1 else 1.0
            for idx in nonzero_idx:
                stim_events.append(
                    StimEvent(
                        name=cond_name,
                        onset=float(time_vector[idx]),
                        duration=dt,
                        value=float(col[idx]),
                    )
                )

    # --- metadata ----------------------------------------------------------
    metadata: dict[str, Any] = {}
    # Carry forward any extra top-level fields as freeform metadata
    skip = {"d", "t", "SD", "s", "aux", "__header__", "__version__", "__globals__"}
    for key, val in mat.items():
        if key not in skip:
            try:
                metadata[key] = val.item() if hasattr(val, "item") else val
            except (ValueError, AttributeError):
                metadata[key] = val

    return NirsData(
        data_matrix=data_matrix,
        time_vector=time_vector,
        wavelengths=wavelengths,
        source_pos=source_pos,
        detector_pos=detector_pos,
        meas_list=meas_list,
        stim_events=stim_events,
        metadata=metadata,
    )


def _parse_snirf(path: Path) -> NirsData:
    """Read a SNIRF 1.1 HDF5 file and return a ``NirsData`` pivot object.

    Only ``/nirs/data1`` is read.  Additional data blocks emit a UserWarning.
    Fields with no .nirs equivalent are discarded with a UserWarning.
    """
    try:
        f = h5py.File(str(path), "r")
    except Exception as exc:
        raise SnirfParseError(f"Cannot open .snirf file: {exc}", path=path) from exc

    with f:
        # S-002: Reject files with external links
        _check_external_links(f, path)

        # --- check for extra data blocks -----------------------------------
        nirs = f.get("nirs")
        if nirs is None:
            raise SnirfParseError("Root group '/nirs' not found.", path=path)

        extra_data = [k for k in nirs if k.startswith("data") and k != "data1"]
        if extra_data:
            warnings.warn(
                f"'{path.name}' contains additional data blocks "
                f"({', '.join('/nirs/' + k for k in extra_data)}). "
                "Only '/nirs/data1' will be read.",
                UserWarning,
                stacklevel=3,
            )

        data1 = nirs.get("data1")
        if data1 is None:
            raise SnirfParseError("Required group '/nirs/data1' not found.", path=path)

        # --- data matrix ---------------------------------------------------
        if "dataTimeSeries" not in data1:
            raise SnirfParseError(
                "Required dataset '/nirs/data1/dataTimeSeries' not found.", path=path
            )
        # S-001: Check shape before allocating
        dts = data1["dataTimeSeries"]
        _validate_array_shape(dts.shape, "dataTimeSeries", path)

        data_matrix = np.array(dts, dtype=np.float64)
        if data_matrix.ndim == 1:
            data_matrix = data_matrix.reshape(-1, 1)

        # --- time vector ---------------------------------------------------
        if "time" not in data1:
            raise SnirfParseError("Required dataset '/nirs/data1/time' not found.", path=path)
        time_raw = np.array(data1["time"], dtype=np.float64).ravel()
        # SNIRF allows (t0, dt) compact form — expand if needed
        if time_raw.size == 2 and data_matrix.shape[0] > 2:
            t0, dt = float(time_raw[0]), float(time_raw[1])
            time_vector = np.arange(data_matrix.shape[0], dtype=np.float64) * dt + t0
        else:
            time_vector = time_raw

        # --- measurement list ----------------------------------------------
        meas_list: list[MeasurementChannel] = []
        ml_idx = 1
        while True:
            ml_key = f"measurementList{ml_idx}"
            if ml_key not in data1:
                break
            ml = data1[ml_key]
            src = int(np.array(ml["sourceIndex"]).ravel()[0])
            det = int(np.array(ml["detectorIndex"]).ravel()[0])
            wl = int(np.array(ml["wavelengthIndex"]).ravel()[0])
            dt_val = int(np.array(ml["dataType"]).ravel()[0]) if "dataType" in ml else 1
            meas_list.append(
                MeasurementChannel(
                    source_index=src,
                    detector_index=det,
                    wavelength_index=wl,
                    data_type=dt_val,
                )
            )
            ml_idx += 1

        if not meas_list:
            raise SnirfParseError(
                "No measurementList entries found in '/nirs/data1'.", path=path
            )

        # --- probe ---------------------------------------------------------
        probe = nirs.get("probe")
        if probe is None:
            raise SnirfParseError("Required group '/nirs/probe' not found.", path=path)

        if "wavelengths" not in probe:
            raise SnirfParseError(
                "Required dataset '/nirs/probe/wavelengths' not found.", path=path
            )
        wavelengths = [float(w) for w in np.array(probe["wavelengths"]).ravel()]

        # source positions — prefer 3D, fall back to 2D
        if "sourcePos3D" in probe:
            src_pos = np.array(probe["sourcePos3D"], dtype=np.float64)
        elif "sourcePos2D" in probe:
            warnings.warn(
                f"'{path.name}': '/nirs/probe/sourcePos3D' not found; "
                "using sourcePos2D with z=0.0.",
                UserWarning,
                stacklevel=3,
            )
            src_raw = np.array(probe["sourcePos2D"], dtype=np.float64)
            src_pos = np.hstack([src_raw, np.zeros((src_raw.shape[0], 1))])
        else:
            raise SnirfParseError(
                "Neither 'sourcePos3D' nor 'sourcePos2D' found in '/nirs/probe'.", path=path
            )

        if "detectorPos3D" in probe:
            det_pos = np.array(probe["detectorPos3D"], dtype=np.float64)
        elif "detectorPos2D" in probe:
            warnings.warn(
                f"'{path.name}': '/nirs/probe/detectorPos3D' not found; "
                "using detectorPos2D with z=0.0.",
                UserWarning,
                stacklevel=3,
            )
            det_raw = np.array(probe["detectorPos2D"], dtype=np.float64)
            det_pos = np.hstack([det_raw, np.zeros((det_raw.shape[0], 1))])
        else:
            raise SnirfParseError(
                "Neither 'detectorPos3D' nor 'detectorPos2D' found in '/nirs/probe'.",
                path=path,
            )

        # warn about SNIRF-only probe fields
        snirf_only_probe = {"coordinateSystem", "coordinateSystemDescription", "landmarkPos3D",
                            "landmarkLabels", "frequencies", "timeDelays", "timeDelayWidths",
                            "momentOrders", "correlationTimeDelays", "correlationTimeDelayWidths"}
        dropped_probe = snirf_only_probe.intersection(set(probe.keys()))
        if dropped_probe:
            warnings.warn(
                f"'{path.name}': The following probe fields have no .nirs equivalent and "
                f"will be discarded: {sorted(dropped_probe)}.",
                UserWarning,
                stacklevel=3,
            )

        # --- stimuli -------------------------------------------------------
        stim_events: list[StimEvent] = []
        stim_idx = 1
        while True:
            stim_key = f"stim{stim_idx}"
            if stim_key not in nirs:
                break
            stim = nirs[stim_key]
            name_bytes = np.array(stim.get("name", np.bytes_(b""))).ravel()
            if name_bytes.size > 0:
                raw = name_bytes[0]
                stim_name = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
            else:
                stim_name = str(float(stim_idx))

            if "data" in stim:
                stim_data = np.array(stim["data"], dtype=np.float64)
                if stim_data.ndim == 1:
                    stim_data = stim_data.reshape(1, -1)
                for row in stim_data:
                    onset = float(row[0]) if row.size > 0 else 0.0
                    duration = float(row[1]) if row.size > 1 else 1.0
                    value = float(row[2]) if row.size > 2 else 1.0
                    stim_events.append(StimEvent(name=stim_name, onset=onset,
                                                  duration=duration, value=value))
            stim_idx += 1

        # --- metadata ------------------------------------------------------
        metadata: dict[str, Any] = {}
        meta_tags = nirs.get("metaDataTags")
        known_meta = {"SubjectID", "MeasurementDate", "MeasurementTime",
                      "LengthUnit", "TimeUnit", "FrequencyUnit"}
        snirf_only_meta: list[str] = []
        if meta_tags is not None:
            for tag_key in meta_tags:
                val_raw = np.array(meta_tags[tag_key]).ravel()
                if val_raw.size > 0:
                    v = val_raw[0]
                    decoded = v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
                else:
                    decoded = ""
                if tag_key in known_meta:
                    metadata[tag_key] = decoded
                else:
                    snirf_only_meta.append(tag_key)
                    metadata[tag_key] = decoded  # carry as freeform

            if snirf_only_meta:
                warnings.warn(
                    f"'{path.name}': The following metaDataTags have no .nirs equivalent "
                    f"and will not survive a round-trip back to .snirf: "
                    f"{sorted(snirf_only_meta)}.",
                    UserWarning,
                    stacklevel=3,
                )

    return NirsData(
        data_matrix=data_matrix,
        time_vector=time_vector,
        wavelengths=wavelengths,
        source_pos=src_pos,
        detector_pos=det_pos,
        meas_list=meas_list,
        stim_events=stim_events,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Internal helpers — writing
# ---------------------------------------------------------------------------


def _write_snirf(data: NirsData, path: Path) -> None:
    """Write ``NirsData`` to a SNIRF 1.1 HDF5 file at ``path``."""
    now = datetime.now(tz=timezone.utc)
    try:
        with h5py.File(str(path), "w") as f:
            nirs = f.create_group("nirs")

            # --- metaDataTags ----------------------------------------------
            meta = nirs.create_group("metaDataTags")
            _hdf_str(meta, "SubjectID", str(data.metadata.get("SubjectID", "unknown")))
            _hdf_str(meta, "MeasurementDate",
                     str(data.metadata.get("MeasurementDate", now.strftime("%Y-%m-%d"))))
            _hdf_str(meta, "MeasurementTime",
                     str(data.metadata.get("MeasurementTime", now.strftime("%H:%M:%S.") +
                                           f"{now.microsecond // 1000:03d}")))
            _hdf_str(meta, "LengthUnit", str(data.metadata.get("LengthUnit", "mm")))
            _hdf_str(meta, "TimeUnit", str(data.metadata.get("TimeUnit", "s")))
            _hdf_str(meta, "FrequencyUnit", str(data.metadata.get("FrequencyUnit", "Hz")))

            # --- data1 -----------------------------------------------------
            data1 = nirs.create_group("data1")
            data1.create_dataset("dataTimeSeries", data=data.data_matrix.astype(np.float64))
            data1.create_dataset("time", data=data.time_vector.astype(np.float64))

            for i, ch in enumerate(data.meas_list):
                ml = data1.create_group(f"measurementList{i + 1}")
                ml.create_dataset("sourceIndex", data=np.int32(ch.source_index))
                ml.create_dataset("detectorIndex", data=np.int32(ch.detector_index))
                ml.create_dataset("wavelengthIndex", data=np.int32(ch.wavelength_index))
                ml.create_dataset("dataType", data=np.int32(ch.data_type))
                ml.create_dataset("dataTypeIndex", data=np.int32(1))

            # --- probe -----------------------------------------------------
            probe = nirs.create_group("probe")
            probe.create_dataset("wavelengths",
                                  data=np.array(data.wavelengths, dtype=np.float64))
            probe.create_dataset("sourcePos3D",
                                  data=data.source_pos.astype(np.float64))
            probe.create_dataset("detectorPos3D",
                                  data=data.detector_pos.astype(np.float64))
            # Write 2D projections as well (XY plane)
            probe.create_dataset("sourcePos2D",
                                  data=data.source_pos[:, :2].astype(np.float64))
            probe.create_dataset("detectorPos2D",
                                  data=data.detector_pos[:, :2].astype(np.float64))

            # --- stim ------------------------------------------------------
            by_name: dict[str, list[StimEvent]] = {}
            for ev in data.stim_events:
                by_name.setdefault(ev.name, []).append(ev)
            for stim_idx, (stim_name, events) in enumerate(by_name.items(), start=1):
                sg = nirs.create_group(f"stim{stim_idx}")
                _hdf_str(sg, "name", stim_name)
                stim_matrix = np.array(
                    [[ev.onset, ev.duration, ev.value] for ev in events],
                    dtype=np.float64,
                )
                sg.create_dataset("data", data=stim_matrix)

    except SnirfWriteError:
        raise
    except Exception as exc:
        raise SnirfWriteError(f"Failed to write .snirf file: {exc}", path=path) from exc


def _write_nirs(data: NirsData, path: Path) -> None:
    """Write ``NirsData`` to a HOMER2-compatible .nirs MAT-file (v5)."""
    # Build s matrix (n_samples x n_conditions) from stim events
    n_samples = data.data_matrix.shape[0]
    condition_names = list(dict.fromkeys(ev.name for ev in data.stim_events))
    if condition_names:
        s_matrix = np.zeros((n_samples, len(condition_names)), dtype=np.float64)
        for ev in data.stim_events:
            col = condition_names.index(ev.name)
            # Find nearest sample index to onset
            sample_idx = int(np.argmin(np.abs(data.time_vector - ev.onset)))
            s_matrix[sample_idx, col] = ev.value
    else:
        s_matrix = np.zeros((n_samples, 1), dtype=np.float64)

    # Build MeasList array (n_channels x 4)
    meas_list_arr = np.array(
        [[ch.source_index, ch.detector_index, 1, ch.wavelength_index]
         for ch in data.meas_list],
        dtype=np.float64,
    )

    # Build SD struct as dict (scipy.io.savemat accepts dicts for structs)
    sd_dict = {
        "Lambda": np.array(data.wavelengths, dtype=np.float64),
        "SrcPos": data.source_pos.astype(np.float64),
        "DetPos": data.detector_pos.astype(np.float64),
        "MeasList": meas_list_arr,
        "nSrcs": np.int32(data.source_pos.shape[0]),
        "nDets": np.int32(data.detector_pos.shape[0]),
    }

    mat_dict: dict[str, Any] = {
        "d": data.data_matrix.astype(np.float64),
        "t": data.time_vector.astype(np.float64),
        "s": s_matrix,
        "SD": sd_dict,
    }

    try:
        scipy.io.savemat(str(path), mat_dict, format="5", do_compression=False)
    except Exception as exc:
        raise NirsWriteError(f"Failed to write .nirs file: {exc}", path=path) from exc


def _hdf_str(group: h5py.Group, name: str, value: str) -> None:
    """Write a scalar string dataset to an HDF5 group (SNIRF convention)."""
    group.create_dataset(name, data=np.bytes_(value.encode("utf-8")))
