"""Engine-layer exceptions."""

from __future__ import annotations

from nirspy.domain.exceptions import (
    ConverterError,
    DomainError,
    ExecutionError,
    NirsDataError,
    NirsParseError,
    NirspyError,
    NirsWriteError,
    SnirfParseError,
    SnirfWriteError,
    ValidationError,
)


class EngineError(NirspyError):
    """Base class for engine-layer errors.

    Inherits directly from :class:`~nirspy.domain.exceptions.NirspyError` so
    the GUI can catch all engine errors with ``except NirspyError`` without
    coupling to domain internals.
    """


class MNEOperationError(EngineError):
    """Raised when an MNE/MNE-NIRS call fails.

    Wraps the original MNE exception so it is available for logging while
    presenting a clean engine-level error to callers.
    """

    def __init__(self, message: str, mne_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.mne_exception: Exception | None = mne_exception
        """The original exception raised by MNE/MNE-NIRS, or ``None`` if not applicable."""


class SnirfLoadError(MNEOperationError):
    """Raised when a SNIRF file cannot be loaded or is malformed."""


class AdapterError(EngineError):
    """Raised when the MNE adapter encounters an unexpected state."""


# ---------------------------------------------------------------------------
# Human-friendly error messages for the GUI layer (ADR-018: EN)
# ---------------------------------------------------------------------------

UI_ERROR_MESSAGES: dict[type[NirspyError], str] = {
    # Engine errors
    SnirfLoadError: (
        "Could not load the SNIRF file. "
        "Please check that the file exists and is not corrupted."
    ),
    MNEOperationError: (
        "An error occurred during signal processing. "
        "Try adjusting the block parameters or check the log for details."
    ),
    AdapterError: (
        "An internal processing error occurred. "
        "Please check the log file for details."
    ),
    EngineError: (
        "An internal engine error occurred. "
        "Please check the log file for details."
    ),
    # Domain errors
    ValidationError: (
        "Pipeline validation failed. "
        "Check the type compatibility between consecutive blocks."
    ),
    ExecutionError: (
        "Pipeline execution failed. "
        "Check that input data is valid and block parameters are correct."
    ),
    DomainError: (
        "A domain error occurred. "
        "Please check your pipeline configuration."
    ),
    # Converter errors
    NirsParseError: (
        "Could not read the .nirs file. "
        "Please check that it is a valid HOMER .nirs file."
    ),
    NirsWriteError: (
        "Could not write the .nirs file. "
        "Please check disk space and write permissions."
    ),
    SnirfParseError: (
        "Could not read the .snirf file. "
        "Please check that it is a valid SNIRF (HDF5) file."
    ),
    SnirfWriteError: (
        "Could not write the .snirf file. "
        "Please check disk space and write permissions."
    ),
    NirsDataError: (
        "The data structure is invalid or inconsistent. "
        "Please check that the file contains valid fNIRS data."
    ),
    ConverterError: (
        "File conversion failed. "
        "Please check that the input file is valid and try again."
    ),
    # Catch-all for NirspyError
    NirspyError: (
        "An unexpected error occurred. "
        "Please check the log file for details."
    ),
}


def get_user_message(exc: NirspyError) -> str:
    """Look up a human-friendly message for an exception.

    Walks the exception's MRO to find the most specific entry in
    :data:`UI_ERROR_MESSAGES`.  Falls back to the generic NirspyError
    message if nothing matches.

    Parameters
    ----------
    exc:
        Any NirspyError subclass instance.

    Returns
    -------
    str
        User-facing message in English, no stack trace.
    """
    for cls in type(exc).__mro__:
        if cls in UI_ERROR_MESSAGES:
            return UI_ERROR_MESSAGES[cls]
    return (
        "An unexpected error occurred. "
        "Please check the log file for details."
    )
