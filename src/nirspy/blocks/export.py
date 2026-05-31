"""Export blocks -- v0.4 (T-037).

ExportTableBlock: exports pipeline results (Evoked or DataFrame) to
tabular formats (CSV, Parquet).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError
from nirspy.engine.mne_adapter import MNEAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ExportTable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportTableParams:
    """Parameters for table export.

    Attributes
    ----------
    output_path:
        Directory path for output files. Default "results".
    format:
        Output format: "csv" or "parquet". Default "csv".
    include_metadata:
        Whether to include metadata columns (condition, channel type).
        Default True.
    """

    output_path: str = "results"
    format: str = "csv"
    include_metadata: bool = True


_EXPORT_SPEC = BlockSpec(
    block_id="export_table",
    display_name="Export Table",
    input_type=DataType.ANY,
    output_type=DataType.NONE,
    params_class=ExportTableParams,
    description=(
        "Exports pipeline results to tabular format (CSV or Parquet). "
        "Sink block -- produces no downstream output."
    ),
)


class ExportTableBlock:
    """Export pipeline data to tabular files.

    Accepts:
    - dict[str, mne.Evoked]: converts to DataFrame via adapter, then exports
    - pandas.DataFrame: exports directly

    This is a sink block (output_type=NONE) -- it writes to disk and
    produces no downstream data for subsequent blocks.

    Supported formats:
    - CSV: always available
    - Parquet: requires pyarrow; raises ValidationError if unavailable
    """

    SPEC: ClassVar[BlockSpec] = _EXPORT_SPEC

    def __init__(
        self,
        params: ExportTableParams | None = None,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: ExportTableParams = params or ExportTableParams()
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _EXPORT_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute table export."""
        if not inputs:
            raise ValidationError(
                "ExportTableBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        # Validate format
        if self.params.format not in ("csv", "parquet"):
            raise ValidationError(
                f"ExportTableBlock: unsupported format '{self.params.format}'. "
                f"Supported: 'csv', 'parquet'."
            )

        data = next(iter(inputs.values()))

        # Convert input to DataFrame
        df = self._to_dataframe(data)

        # Ensure output directory exists
        output_dir = Path(self.params.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write file
        filename = f"export.{self.params.format}"
        output_file = output_dir / filename

        if self.params.format == "csv":
            df.to_csv(output_file, index=True)
        elif self.params.format == "parquet":
            self._write_parquet(df, output_file)

        metadata: dict[str, Any] = {
            "output_file": str(output_file),
            "format": self.params.format,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
        }

        return BlockResult(
            data=None,
            block_id=_EXPORT_SPEC.block_id,
            metadata=metadata,
        )

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert input data to a pandas DataFrame.

        Supports:
        - dict[str, mne.Evoked]: uses adapter.evoked_to_dataframe
        - pd.DataFrame: passthrough
        """
        if isinstance(data, pd.DataFrame):
            return data

        # Check if it's a dict of Evoked objects
        if isinstance(data, dict):
            import mne

            # Verify all values are Evoked
            sample = next(iter(data.values()), None)
            if sample is not None and isinstance(sample, mne.Evoked):
                return self._adapter.evoked_to_dataframe(data)

        raise ValidationError(
            f"ExportTableBlock: unsupported input type '{type(data).__name__}'. "
            f"Expected dict[str, Evoked] or pandas.DataFrame."
        )

    @staticmethod
    def _write_parquet(df: pd.DataFrame, path: Path) -> None:
        """Write DataFrame to Parquet with graceful fallback."""
        try:
            import pyarrow  # noqa: F401

            df.to_parquet(path, index=True)
        except ImportError:
            raise ValidationError(
                "ExportTableBlock: Parquet export requires 'pyarrow'. "
                "Install with: pip install pyarrow"
            ) from None
