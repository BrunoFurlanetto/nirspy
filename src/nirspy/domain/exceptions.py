"""Domain-layer exceptions — no external dependencies."""

from __future__ import annotations

from pathlib import Path


class NirspyError(Exception):
    """Root of the nirspy exception hierarchy.

    GUI code can use ``except NirspyError`` to catch all package errors
    without coupling to specific domain or engine subclasses.
    """


class DomainError(NirspyError):
    """Base class for all domain-layer errors."""


class ValidationError(DomainError):
    """Raised when pipeline or block configuration is invalid."""


class ExecutionError(DomainError):
    """Raised when pipeline execution fails."""


# ---------------------------------------------------------------------------
# Converter exceptions (T-002)
# ---------------------------------------------------------------------------


class ConverterError(NirspyError):
    """Root of the .nirs <-> .snirf converter exception hierarchy.

    GUI / CLI can use ``except ConverterError`` to handle all converter
    failures without importing ``nirspy.io.converters`` directly.
    """


class NirsParseError(ConverterError):
    """Raised when a .nirs MAT-file cannot be read or has an invalid structure."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


class NirsWriteError(ConverterError):
    """Raised when writing a .nirs MAT-file fails."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


class SnirfParseError(ConverterError):
    """Raised when a .snirf HDF5 file is invalid or missing required fields."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


class SnirfWriteError(ConverterError):
    """Raised when writing a .snirf HDF5 file fails."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


class NirsDataError(ConverterError):
    """Raised when the NirsData pivot representation violates its invariants."""
