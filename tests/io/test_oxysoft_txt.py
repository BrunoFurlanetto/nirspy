"""Tests for the Oxysoft .txt → .snirf converter."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np
import pytest

from nirspy.domain.exceptions import ConverterError, NirsParseError
from nirspy.io import oxysoft_txt_to_snirf
from nirspy.io.oxysoft_txt import _parse_header, _parse_oxysoft_txt


def _minimal_txt(n_samples: int = 5, with_events: bool = True) -> str:
    """Build the smallest valid Oxysoft .txt body for the parser."""
    rows = []
    for i in range(n_samples):
        ev = "S1" if (with_events and i == 2) else "0"
        rows.append(
            f"  {i}\t{1.0 + i:.4f}\t{2.0 + i:.4f}\t"
            f"{3.0 + i:.4f}\t{4.0 + i:.4f}\t{ev}"
        )
    body = "\n".join(rows)
    return dedent(
        f"""\
        Oxysoft export of:\tdummy
        Datafile sample rate:\t10.00\tHz
        Datafile duration:\t{n_samples / 10:.2f}\ts
        Datafile total number of samples:\t{n_samples}

        Optode distance (mm):\t 35.00\t35.00
        DPF:\t6.61\t6.61

        Light source wavelengths:
        device\tindex\twavelength
        1\t1\t840\tnm
        1\t2\t752\tnm

        Legend:
        Column\tTrace (Measurement)
        1\t(Sample number)
        2\tRx1 - Tx1 O2Hb (s)
        3\tRx1 - Tx1 HHb (s)
        4\tRx1 - Tx2 O2Hb (s)
        5\tRx1 - Tx2 HHb (s)
        6\t(Event)


        1\t2\t3\t4\t5\t6
        {body}
        """
    )


def test_parse_header_extracts_sample_rate_and_wavelengths() -> None:
    header = _parse_header(_minimal_txt())
    assert header["sample_rate"] == 10.0
    assert header["wavelengths"] == {1: 840.0, 2: 752.0}
    assert header["dpf"] == [6.61, 6.61]
    assert header["distance_mm"] == [35.0, 35.0]


def test_parse_header_collects_channel_legend() -> None:
    header = _parse_header(_minimal_txt())
    assert len(header["channels"]) == 4
    assert header["channels"][0] == {
        "col": 2,
        "rx": 1,
        "tx": 1,
        "species": "O2Hb",
    }


def test_parse_oxysoft_txt_returns_valid_nirsdata(tmp_path: Path) -> None:
    path = tmp_path / "rec.txt"
    path.write_text(_minimal_txt(n_samples=10))

    data = _parse_oxysoft_txt(path)

    assert data.data_matrix.shape == (10, 4)
    assert data.time_vector.shape == (10,)
    assert data.wavelengths == [752.0, 840.0]
    assert len(data.meas_list) == 4
    assert len(data.stim_events) == 1
    assert data.stim_events[0].name == "S1"


def test_oxysoft_txt_to_snirf_writes_readable_file(tmp_path: Path) -> None:
    in_path = tmp_path / "rec.txt"
    out_path = tmp_path / "rec.snirf"
    in_path.write_text(_minimal_txt(n_samples=20))

    oxysoft_txt_to_snirf(in_path, out_path)

    assert out_path.exists()
    with h5py.File(out_path, "r") as f:
        nirs = f["nirs"]
        data = np.array(nirs["data1"]["dataTimeSeries"])
        assert data.shape == (20, 4)
        stim_names = []
        for k in nirs:
            if k.startswith("stim"):
                stim_names.append(
                    bytes(np.array(nirs[k]["name"])).decode("utf-8")
                )
        assert "S1" in stim_names


def test_overwrite_guard_raises_when_output_exists(tmp_path: Path) -> None:
    in_path = tmp_path / "rec.txt"
    out_path = tmp_path / "out.snirf"
    in_path.write_text(_minimal_txt())
    out_path.write_bytes(b"existing")

    with pytest.raises(ConverterError):
        oxysoft_txt_to_snirf(in_path, out_path, overwrite=False)


def test_overwrite_true_replaces_existing(tmp_path: Path) -> None:
    in_path = tmp_path / "rec.txt"
    out_path = tmp_path / "out.snirf"
    in_path.write_text(_minimal_txt())
    out_path.write_bytes(b"existing")

    oxysoft_txt_to_snirf(in_path, out_path, overwrite=True)

    with h5py.File(out_path, "r") as f:
        assert "nirs" in f


def test_missing_sample_rate_raises(tmp_path: Path) -> None:
    in_path = tmp_path / "rec.txt"
    in_path.write_text("Oxysoft export of:\tdummy\n\nLegend:\n1\t(Sample)\n")

    with pytest.raises(NirsParseError):
        oxysoft_txt_to_snirf(in_path, in_path.with_suffix(".snirf"))


def test_unknown_extension_rejected(tmp_path: Path) -> None:
    in_path = tmp_path / "rec.nirs"
    in_path.write_text(_minimal_txt())

    with pytest.raises(ConverterError):
        oxysoft_txt_to_snirf(
            in_path, tmp_path / "out.snirf"
        )
