"""Domain-layer exceptions — no external dependencies."""

from __future__ import annotations


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
