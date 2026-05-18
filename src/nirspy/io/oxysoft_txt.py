"""Converter from Oxysoft .txt export to .snirf.

The Artinis Oxysoft software exports a tab-separated text dump that contains
the recording header, per-channel raw intensity samples and an event column.
This converter parses that file and emits a SNIRF 1.1 HDF5 with
``dataType = 1`` (continuous-wave amplitude) so downstream NIRSPY blocks
(``OpticalDensity`` → ``BeerLambert`` → ...) can run normally.

Important: the column legend in the .txt labels traces as ``O2Hb`` / ``HHb``,
but the numerical data is **raw light intensity** (Oxysoft's labelling
quirk). Treating it as concentration would break the pipeline.

Event labels — unlike the Artinis ``.oxy3 → .nirs`` round-trip which loses
them — are preserved verbatim from column 18 of the data block.

Future work (TODO): direct ``.oxy3 → .snirf`` once an open-source parser for
the proprietary binary format becomes available. Until then this .txt route
is the only way to keep semantic event names.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from nirspy.domain.exceptions import (
    NirsDataError,
    NirsParseError,
)
from nirspy.io.converters import (
    MAX_CHANNELS,
    MAX_SAMPLES,
    MeasurementChannel,
    NirsData,
    StimEvent,
    _atomic_create_output,
    _strip_pii_from_metadata,
    _validate_input,
    _write_snirf,
)

__all__ = ["oxysoft_txt_to_snirf"]


# Channel legend lines look like: "2\tRx1 - Tx1 O2Hb (dp372)"
_LEGEND_RE = re.compile(
    r"^\s*(?P<col>\d+)\s+Rx(?P<rx>\d+)\s*-\s*Tx(?P<tx>\d+)\s+(?P<species>\S+)"
)


def _parse_header(text: str) -> dict[str, Any]:
    """Extract sample rate, wavelengths, DPF, distances and channel legend."""
    header: dict[str, Any] = {
        "sample_rate": None,
        "n_samples": None,
        "duration": None,
        "dpf": [],
        "distance_mm": [],
        "wavelengths": {},  # source_idx -> wavelength_nm
        "channels": [],  # list[dict] in column order
        "subject_id": None,
        "measurement_date": None,
        "measurement_time": None,
    }

    in_legend = False
    in_wavelengths = False
    for raw_line in text.splitlines():
        line = raw_line.lstrip(" ")

        if line.startswith("Datafile sample rate:"):
            header["sample_rate"] = float(line.split("\t")[1])
        elif line.startswith("Datafile duration:"):
            header["duration"] = float(line.split("\t")[1])
        elif line.startswith("Datafile total number of samples:"):
            header["n_samples"] = int(line.split("\t")[1])
        elif line.startswith("Optode distance (mm):"):
            parts = line.split("\t")[1:]
            header["distance_mm"] = [float(p) for p in parts if p.strip()]
        elif line.startswith("DPF:"):
            parts = line.split("\t")[1:]
            header["dpf"] = [float(p) for p in parts if p.strip()]
        elif line.startswith("Start of measurement:"):
            stamp = line.split("\t", 1)[1].strip()
            try:
                dt = datetime.fromisoformat(stamp.replace(" ", "T"))
            except ValueError:
                continue
            header["measurement_date"] = dt.strftime("%Y-%m-%d")
            header["measurement_time"] = dt.strftime("%H:%M:%S")
        elif line.startswith("Light source wavelengths:"):
            in_wavelengths = True
            continue
        elif line.startswith("Legend:"):
            in_wavelengths = False
            in_legend = True
            continue
        elif in_wavelengths:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
                src_idx = int(parts[1])
                wl_nm = float(parts[2])
                header["wavelengths"][src_idx] = wl_nm
        elif in_legend:
            m = _LEGEND_RE.match(line)
            if m:
                header["channels"].append(
                    {
                        "col": int(m.group("col")),
                        "rx": int(m.group("rx")),
                        "tx": int(m.group("tx")),
                        "species": m.group("species"),
                    }
                )
            elif line.startswith("Column") or not line.strip():
                continue
            else:
                # Plain numeric header row signals start of data
                if all(tok.isdigit() for tok in line.split()):
                    break
    return header


def _parse_data_block(
    text: str, n_data_cols: int
) -> tuple[np.ndarray[Any, Any], list[tuple[int, str]]]:
    """Return ``(intensity_matrix, events)`` from the tab-delimited data block.

    ``intensity_matrix`` has shape ``(n_samples, n_data_cols)``. Events are
    ``(sample_index, label)`` taken from the last column.
    """
    samples: list[list[float]] = []
    events: list[tuple[int, str]] = []
    seen_numeric_header = False
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        tokens = raw_line.split()
        if not tokens:
            continue
        if (
            tokens[0].isdigit()
            and len(tokens) >= 2
            and not seen_numeric_header
            and all(t.isdigit() for t in tokens)
        ):
            seen_numeric_header = True
            continue
        if not seen_numeric_header:
            continue
        # Data row: sample_idx, then n_data_cols numbers, then maybe event
        try:
            sample_idx = int(tokens[0])
        except ValueError:
            continue
        body = tokens[1:]
        if len(body) < n_data_cols:
            continue
        values_str = body[:n_data_cols]
        try:
            values = [float(v) for v in values_str]
        except ValueError:
            continue
        samples.append(values)
        if len(body) > n_data_cols:
            label = body[n_data_cols].strip()
            if label and label != "0":
                events.append((sample_idx, label))

    if not samples:
        raise NirsParseError("Oxysoft .txt file has no numeric data rows.")
    return np.asarray(samples, dtype=np.float64), events


def _parse_oxysoft_txt(path: Path) -> NirsData:
    """Parse an Oxysoft .txt export into the canonical :class:`NirsData`."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise NirsParseError(
            f"Cannot read Oxysoft .txt file: {exc}", path=path
        ) from exc

    header = _parse_header(text)
    if header["sample_rate"] is None:
        raise NirsParseError(
            "Oxysoft .txt header missing 'Datafile sample rate'.", path=path
        )
    if not header["wavelengths"]:
        raise NirsParseError(
            "Oxysoft .txt header missing 'Light source wavelengths' table.",
            path=path,
        )
    if not header["channels"]:
        # Common when Oxysoft export deselected the data traces (legend lists
        # only "Sample number" + "(Event)" and the data block has no channel
        # columns). Without channel data we cannot produce a usable SNIRF.
        raise NirsParseError(
            "Oxysoft .txt export contains no channel data — only events were "
            "exported. Re-export from Oxysoft with the optical traces "
            "selected (Rx - Tx columns) before converting.",
            path=path,
        )

    n_data_cols = len(header["channels"])

    data_matrix, raw_events = _parse_data_block(text, n_data_cols)

    n_samples, n_cols = data_matrix.shape
    if n_samples > MAX_SAMPLES:
        raise NirsDataError(
            f"Oxysoft .txt exceeds MAX_SAMPLES ({MAX_SAMPLES}): got {n_samples}."
        )
    if n_cols > MAX_CHANNELS:
        raise NirsDataError(
            f"Oxysoft .txt exceeds MAX_CHANNELS ({MAX_CHANNELS}): got {n_cols}."
        )

    sample_rate = header["sample_rate"]
    time_vector = np.arange(n_samples, dtype=np.float64) / float(sample_rate)

    # Build unique wavelength list (sorted, deterministic)
    wl_set = sorted({float(v) for v in header["wavelengths"].values()})
    if not wl_set:
        raise NirsParseError("No wavelengths found in header.", path=path)
    wl_index_of: dict[float, int] = {wl: i + 1 for i, wl in enumerate(wl_set)}

    # Map each channel column → (source_idx, detector_idx, wavelength_idx)
    # The .txt source is the Tx number; species (O2Hb/HHb) maps to a wavelength
    # by looking at the source's listed wavelength in the header.
    src_seen: dict[int, int] = {}  # tx -> 1-based source index in SNIRF
    det_seen: dict[int, int] = {}  # rx -> 1-based detector index in SNIRF
    meas_list: list[MeasurementChannel] = []

    species_to_wl_hint = {"O2Hb": "long", "HHb": "short"}

    long_wl = max(wl_set)
    short_wl = min(wl_set)

    for ch in header["channels"]:
        tx = ch["tx"]
        rx = ch["rx"]
        species = ch["species"]
        if tx not in src_seen:
            src_seen[tx] = len(src_seen) + 1
        if rx not in det_seen:
            det_seen[rx] = len(det_seen) + 1
        # Heuristic: Oxysoft labels the longer wavelength as O2Hb, shorter as HHb
        which = species_to_wl_hint.get(species, "long")
        chosen_wl = long_wl if which == "long" else short_wl
        wl_idx = wl_index_of[chosen_wl]
        meas_list.append(
            MeasurementChannel(
                source_index=src_seen[tx],
                detector_index=det_seen[rx],
                wavelength_index=wl_idx,
                data_type=1,
            )
        )

    n_sources = len(src_seen)
    n_detectors = len(det_seen)

    # Placeholder optode positions — Oxysoft .txt header has only optode
    # distances, not 3D coordinates. Spread sources along x and put detectors
    # at the origin/offset. Downstream pipeline (OD, mBLL, filtering) does not
    # need true positions; viewers degrade gracefully (probe viewer fallback).
    src_pos = np.zeros((n_sources, 3), dtype=np.float64)
    det_pos = np.zeros((n_detectors, 3), dtype=np.float64)
    distance = (
        float(header["distance_mm"][0])
        if header["distance_mm"]
        else 30.0
    )
    for i in range(n_sources):
        src_pos[i, 0] = float(i) * distance
    for i in range(n_detectors):
        det_pos[i, 0] = float(i) * distance + distance / 2.0

    # Stim events keep their original labels
    stim_events: list[StimEvent] = []
    sfreq_period = 1.0 / float(sample_rate)
    for sample_idx, label in raw_events:
        if 0 <= sample_idx < n_samples:
            stim_events.append(
                StimEvent(
                    name=label,
                    onset=float(time_vector[sample_idx]),
                    duration=sfreq_period,
                    value=1.0,
                )
            )

    metadata: dict[str, Any] = {
        "SourceFormat": "Oxysoft-txt",
        "DPF": header["dpf"],
        "OptodeDistanceMM": header["distance_mm"],
    }
    if header["measurement_date"]:
        metadata["MeasurementDate"] = header["measurement_date"]
    if header["measurement_time"]:
        metadata["MeasurementTime"] = header["measurement_time"]

    data = NirsData(
        data_matrix=data_matrix,
        time_vector=time_vector,
        wavelengths=wl_set,
        source_pos=src_pos,
        detector_pos=det_pos,
        meas_list=meas_list,
        stim_events=stim_events,
        metadata=metadata,
    )
    data.validate()
    return data


def oxysoft_txt_to_snirf(
    input_path: Path | str,
    output_path: Path | str,
    overwrite: bool = False,
    *,
    strip_pii: bool = False,
) -> None:
    """Convert an Oxysoft .txt export to a SNIRF 1.1 file.

    Args:
        input_path:  Path to the .txt input file.
        output_path: Path for the .snirf output file.
        overwrite:   If ``False`` (default), raises ``ConverterError`` when
                     ``output_path`` already exists.
        strip_pii:   When ``True``, common PII fields are removed before write.

    Raises:
        ConverterError:  ``input_path`` missing, wrong extension, or output
                         path collision with ``overwrite=False``.
        NirsParseError:  Header or data block could not be parsed.
        NirsDataError:   Resulting pivot violates SNIRF size invariants.
        SnirfWriteError: Writing the .snirf file failed.

    Notes:
        Direct ``.oxy3 → .snirf`` is not yet supported because the Oxysoft
        binary format is proprietary. Export to .txt from Oxysoft and use
        this path until an open-source ``.oxy3`` parser is available.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    _validate_input(in_path, ".txt")
    _atomic_create_output(out_path, overwrite)

    data = _parse_oxysoft_txt(in_path)
    if strip_pii:
        data.metadata = _strip_pii_from_metadata(data.metadata)
    _write_snirf(data, out_path)
