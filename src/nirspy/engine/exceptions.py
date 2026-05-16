"""Engine-layer exceptions."""

from __future__ import annotations

from nirspy.domain.exceptions import NirspyError


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
# Human-friendly error messages for the GUI layer
# ---------------------------------------------------------------------------

UI_ERROR_MESSAGES: dict[type[NirspyError], str] = {
    AdapterError: "Erro ao carregar dados. Verifique o arquivo SNIRF.",
    MNEOperationError: "Erro no processamento fNIRS. Veja o log para detalhes.",
    SnirfLoadError: "Arquivo SNIRF inválido ou corrompido. Verifique o caminho e o formato.",
    EngineError: "Erro interno do processamento. Consulte o log para detalhes.",
}
