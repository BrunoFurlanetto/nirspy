"""nirspy.io — pipeline serialisation, converters and file I/O utilities."""

from nirspy.domain.exceptions import (
    ConverterError,
    NirsDataError,
    NirsParseError,
    NirsWriteError,
    SnirfParseError,
    SnirfWriteError,
)
from nirspy.io.converters import (
    MeasurementChannel,
    NirsData,
    StimEvent,
    nirs_to_snirf,
    snirf_to_nirs,
)
from nirspy.io.pipeline_runner import RunResult, run_pipeline
from nirspy.io.yaml_serializer import dump_pipeline, load_pipeline

__all__ = [
    # pipeline serialisation
    "dump_pipeline",
    "load_pipeline",
    # pipeline runner
    "RunResult",
    "run_pipeline",
    # converters
    "nirs_to_snirf",
    "snirf_to_nirs",
    # pivot types
    "MeasurementChannel",
    "NirsData",
    "StimEvent",
    # exceptions
    "ConverterError",
    "NirsDataError",
    "NirsParseError",
    "NirsWriteError",
    "SnirfParseError",
    "SnirfWriteError",
]
