"""Domain-layer exceptions — no external dependencies."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class ValidationError(DomainError):
    """Raised when pipeline or block configuration is invalid."""


class ExecutionError(DomainError):
    """Raised when pipeline execution fails."""
