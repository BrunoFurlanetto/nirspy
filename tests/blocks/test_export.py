"""Tests for ExportTableBlock (T-037)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest

from nirspy.blocks.export import (
    ExportTableBlock,
    ExportTableParams,
)
from nirspy.domain.block import BlockResult
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError


@pytest.fixture()
def evoked_dict() -> dict[str, mne.Evoked]:
    """Create a synthetic dict of Evoked objects."""
    sfreq = 10.0
    n_times = 50
    n_channels = 2

    ch_names = ["S1_D1 hbo", "S1_D1 hbr"]
    ch_types = ["hbo", "hbr"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    for ch in info["chs"]:
        ch["loc"][3:6] = [0.0, 0.0, 0.0]
        ch["loc"][6:9] = [0.03, 0.0, 0.0]

    rng = np.random.default_rng(42)
    data_a = rng.standard_normal((n_channels, n_times)) * 1e-6
    data_b = rng.standard_normal((n_channels, n_times)) * 1e-6

    evoked_a = mne.EvokedArray(data_a, info, tmin=-0.5, verbose=False)
    evoked_b = mne.EvokedArray(data_b, info, tmin=-0.5, verbose=False)

    return {"stim_A": evoked_a, "stim_B": evoked_b}


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for direct export."""
    return pd.DataFrame({
        "time": [0.0, 0.1, 0.2],
        "channel": ["S1_D1 hbo"] * 3,
        "value": [1e-6, 2e-6, 3e-6],
        "condition": ["stim_A"] * 3,
    })


class TestExportTableBlock:
    """Tests for ExportTableBlock."""

    def test_params_defaults(self) -> None:
        """Default params are correct."""
        params = ExportTableParams()
        assert params.output_path == "results"
        assert params.format == "csv"
        assert params.include_metadata is True

    def test_params_frozen(self) -> None:
        """Params are immutable."""
        params = ExportTableParams()
        with pytest.raises(AttributeError):
            params.format = "parquet"  # type: ignore[misc]

    def test_spec_correct(self) -> None:
        """Block spec has correct input/output types."""
        block = ExportTableBlock()
        assert block.spec.block_id == "export_table"
        assert block.spec.input_type == DataType.ANY
        assert block.spec.output_type == DataType.NONE

    def test_raises_on_empty_inputs(self) -> None:
        """Block raises ValidationError when inputs is empty."""
        block = ExportTableBlock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(None, {})

    def test_raises_on_invalid_format(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Block raises ValidationError for unsupported format."""
        params = ExportTableParams(format="xlsx")
        block = ExportTableBlock(params=params)
        with pytest.raises(ValidationError, match="unsupported format"):
            block.run(None, {"prev": sample_dataframe})

    def test_raises_on_unsupported_input_type(self) -> None:
        """Block raises ValidationError for non-convertible input."""
        block = ExportTableBlock()
        with pytest.raises(ValidationError, match="unsupported input type"):
            block.run(None, {"prev": "not a dataframe"})

    def test_export_csv_from_dataframe(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Block exports DataFrame to CSV."""
        params = ExportTableParams(
            output_path=str(tmp_path / "output"),
            format="csv",
        )
        block = ExportTableBlock(params=params)
        result = block.run(None, {"prev": sample_dataframe})

        assert isinstance(result, BlockResult)
        assert result.block_id == "export_table"
        assert result.data is None  # sink block
        assert result.metadata["format"] == "csv"
        assert result.metadata["n_rows"] == 3

        # Verify file was created
        output_file = Path(result.metadata["output_file"])
        assert output_file.exists()
        assert output_file.suffix == ".csv"

    def test_export_csv_from_evoked(
        self, evoked_dict: dict[str, mne.Evoked], tmp_path: Path
    ) -> None:
        """Block converts Evoked dict to DataFrame and exports CSV."""
        params = ExportTableParams(
            output_path=str(tmp_path / "output"),
            format="csv",
        )
        block = ExportTableBlock(params=params)
        result = block.run(None, {"prev": evoked_dict})

        assert result.metadata["format"] == "csv"
        assert result.metadata["n_rows"] > 0
        assert "condition" in result.metadata["columns"]

        output_file = Path(result.metadata["output_file"])
        assert output_file.exists()

        # Read back and verify content
        df = pd.read_csv(output_file)
        assert "condition" in df.columns
        assert set(df["condition"].unique()) == {"stim_A", "stim_B"}

    def test_export_parquet_missing_pyarrow(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Block raises ValidationError when pyarrow is not available."""
        params = ExportTableParams(
            output_path=str(tmp_path / "output"),
            format="parquet",
        )
        block = ExportTableBlock(params=params)

        with (
            patch.dict("sys.modules", {"pyarrow": None}),
            patch("pandas.DataFrame.to_parquet", side_effect=ImportError),
            pytest.raises(ValidationError, match="pyarrow"),
        ):
            block.run(None, {"prev": sample_dataframe})

    def test_export_parquet_success(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Block exports DataFrame to Parquet when pyarrow is available."""
        pytest.importorskip("pyarrow")

        params = ExportTableParams(
            output_path=str(tmp_path / "output"),
            format="parquet",
        )
        block = ExportTableBlock(params=params)
        result = block.run(None, {"prev": sample_dataframe})

        assert result.metadata["format"] == "parquet"
        output_file = Path(result.metadata["output_file"])
        assert output_file.exists()
        assert output_file.suffix == ".parquet"

    def test_registered_in_registry(self) -> None:
        """Block is registered in the global registry."""
        from nirspy.blocks import registry

        assert "export_table" in registry.list_blocks()
