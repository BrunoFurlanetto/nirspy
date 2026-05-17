"""Tests for UI_ERROR_MESSAGES dict defined in nirspy.engine.exceptions.

Verifies that:
- The dict covers the main catchable exception types.
- Every value is a non-empty human-readable string.
- More-specific exceptions are present (SnirfLoadError before EngineError) so
  GUI dispatch can match the most specific message first.
"""

from __future__ import annotations

from nirspy.domain.exceptions import NirspyError
from nirspy.engine.exceptions import (
    UI_ERROR_MESSAGES,
    AdapterError,
    EngineError,
    MNEOperationError,
    SnirfLoadError,
)


class TestUIErrorMessagesCoverage:
    """UI_ERROR_MESSAGES covers the main exception classes."""

    def test_dict_is_not_empty(self) -> None:
        assert len(UI_ERROR_MESSAGES) > 0

    def test_covers_engine_error(self) -> None:
        assert EngineError in UI_ERROR_MESSAGES

    def test_covers_mne_operation_error(self) -> None:
        assert MNEOperationError in UI_ERROR_MESSAGES

    def test_covers_snirf_load_error(self) -> None:
        assert SnirfLoadError in UI_ERROR_MESSAGES

    def test_covers_adapter_error(self) -> None:
        assert AdapterError in UI_ERROR_MESSAGES

    def test_all_values_are_non_empty_strings(self) -> None:
        for exc_type, message in UI_ERROR_MESSAGES.items():
            assert isinstance(message, str), f"{exc_type.__name__} has non-string message"
            assert message.strip(), f"{exc_type.__name__} has blank message"

    def test_keys_are_nirspy_error_subclasses(self) -> None:
        for exc_type in UI_ERROR_MESSAGES:
            assert issubclass(exc_type, NirspyError), (
                f"{exc_type.__name__} is not a NirspyError subclass"
            )


class TestUIErrorMessagesLookup:
    """UI dispatch pattern: find the most specific message for a raised exception."""

    def test_snirf_load_error_resolved_before_engine_error(self) -> None:
        """GUI should display the SnirfLoadError message, not the generic EngineError one."""
        exc = SnirfLoadError("bad file")
        # Simulate GUI dispatch: iterate MRO and pick first match in dict.
        message = _resolve_ui_message(exc)
        assert message == UI_ERROR_MESSAGES[SnirfLoadError]

    def test_mne_operation_error_resolved_before_engine_error(self) -> None:
        exc = MNEOperationError("mne crash")
        message = _resolve_ui_message(exc)
        assert message == UI_ERROR_MESSAGES[MNEOperationError]

    def test_fallback_engine_error_message_for_unknown_subclass(self) -> None:
        """An EngineError subclass not in the dict falls back to EngineError message."""

        class _UnknownEngineError(EngineError):
            pass

        exc = _UnknownEngineError("unexpected")
        message = _resolve_ui_message(exc)
        assert message == UI_ERROR_MESSAGES[EngineError]

    def test_returns_none_for_unregistered_error_type(self) -> None:
        """A plain NirspyError without a mapping returns None from the resolver."""
        exc = NirspyError("base")
        message = _resolve_ui_message(exc)
        assert message is None


# ---------------------------------------------------------------------------
# Helper — simulate GUI dispatch
# ---------------------------------------------------------------------------


def _resolve_ui_message(exc: Exception) -> str | None:
    """Return the most specific UI_ERROR_MESSAGES entry for *exc*, or None."""
    for cls in type(exc).__mro__:
        if cls in UI_ERROR_MESSAGES:
            return UI_ERROR_MESSAGES[cls]  # type: ignore[index]
    return None
