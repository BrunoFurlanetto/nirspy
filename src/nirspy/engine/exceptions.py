"""Engine-layer exceptions."""

from __future__ import annotations

from nirspy.domain.exceptions import DomainError


class EngineError(DomainError):
    """Base class for engine-layer errors."""


class SnirfLoadError(EngineError):
    """Raised when a SNIRF file cannot be loaded or is malformed."""


class AdapterError(EngineError):
    """Raised when the MNE adapter encounters an unexpected state."""
