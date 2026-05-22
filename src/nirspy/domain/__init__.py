"""nirspy.domain — pure domain layer (stdlib + typing only)."""

from nirspy.domain.block import Block, BlockResult, BlockSpec
from nirspy.domain.cache import CacheProtocol
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import DomainError, ExecutionError, NirspyError, ValidationError
from nirspy.domain.execution import (
    ExecutionContext,
    PipelineRunner,
    ProgressCallback,
    run_pipeline_sync,
)
from nirspy.domain.pipeline import Pipeline, RegistryProtocol
from nirspy.domain.validation import validate_io_chain

__all__ = [
    "Block",
    "BlockResult",
    "BlockSpec",
    "CacheProtocol",
    "DataType",
    "DomainError",
    "NirspyError",
    "ExecutionContext",
    "PipelineRunner",
    "ExecutionError",
    "Pipeline",
    "ProgressCallback",
    "RegistryProtocol",
    "ValidationError",
    "run_pipeline_sync",
    "validate_io_chain",
]
