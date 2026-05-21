"""Tests for humanized error messages (T-020).

Validates that all NirspyError subclasses have user-facing messages
in English, and that no stack traces leak to the UI.
"""

from __future__ import annotations

import re

from nirspy.domain.exceptions import NirspyError
from nirspy.engine.exceptions import UI_ERROR_MESSAGES, get_user_message


def _all_nirspy_error_subclasses() -> list[type[NirspyError]]:
    """Recursively collect all concrete NirspyError subclasses."""
    result: list[type[NirspyError]] = []
    queue: list[type[NirspyError]] = [NirspyError]
    while queue:
        cls = queue.pop()
        for sub in cls.__subclasses__():
            result.append(sub)
            queue.append(sub)
    return result


class TestAllNirspyErrorsHaveUIMessage:
    """Every NirspyError subclass must have an entry in UI_ERROR_MESSAGES."""

    def test_all_nirspy_errors_have_ui_message(self) -> None:
        """Introspect subclasses of NirspyError and verify coverage."""
        subclasses = _all_nirspy_error_subclasses()
        assert len(subclasses) > 0, "Expected at least one NirspyError subclass"

        missing: list[str] = []
        for cls in subclasses:
            # get_user_message uses MRO, so it always finds something.
            # But we want direct entries for all classes.
            if cls not in UI_ERROR_MESSAGES:
                missing.append(cls.__name__)

        assert missing == [], (
            f"NirspyError subclasses missing from UI_ERROR_MESSAGES: {missing}"
        )

    def test_nirspy_error_root_has_entry(self) -> None:
        """NirspyError itself has an entry (catch-all)."""
        assert NirspyError in UI_ERROR_MESSAGES


class TestUIMessagesAreEnglish:
    """All UI messages must be in English (ADR-018)."""

    def test_ui_messages_are_ascii(self) -> None:
        """Proxy for EN: no non-ASCII characters in messages."""
        for exc_cls, message in UI_ERROR_MESSAGES.items():
            assert message.isascii(), (
                f"UI message for {exc_cls.__name__} contains "
                f"non-ASCII characters: {message!r}"
            )

    def test_ui_messages_no_portuguese(self) -> None:
        """Messages should not contain common Portuguese words."""
        pt_patterns = [
            r"\bErro\b", r"\berro\b", r"\bVerifique\b", r"\bverifique\b",
            r"\bConsulte\b", r"\bconsulte\b", r"\binválido\b",
            r"\bcorrompido\b", r"\bArquivo\b",
        ]
        for exc_cls, message in UI_ERROR_MESSAGES.items():
            for pattern in pt_patterns:
                assert not re.search(pattern, message), (
                    f"UI message for {exc_cls.__name__} appears to be in "
                    f"Portuguese (matched {pattern!r}): {message!r}"
                )


class TestErrorDisplayNoTraceback:
    """Error messages shown to the user must never contain stack traces."""

    def test_ui_messages_no_traceback(self) -> None:
        """No UI message contains 'Traceback' or 'File \"'."""
        for exc_cls, message in UI_ERROR_MESSAGES.items():
            assert "Traceback" not in message, (
                f"UI message for {exc_cls.__name__} contains 'Traceback'"
            )
            assert 'File "' not in message, (
                f"UI message for {exc_cls.__name__} contains stack trace"
            )


class TestGetUserMessage:
    """get_user_message() resolves the most specific message via MRO."""

    def test_exact_match(self) -> None:
        """Returns the exact entry for a known exception class."""
        from nirspy.engine.exceptions import SnirfLoadError

        exc = SnirfLoadError("test")
        msg = get_user_message(exc)
        assert msg == UI_ERROR_MESSAGES[SnirfLoadError]

    def test_fallback_to_parent(self) -> None:
        """Falls back to parent class if no direct entry."""
        # Create a new subclass not in the registry
        class CustomEngineError(NirspyError):
            pass

        exc = CustomEngineError("test")
        msg = get_user_message(exc)
        # Should fall back to NirspyError entry
        assert msg == UI_ERROR_MESSAGES[NirspyError]

    def test_fallback_generic(self) -> None:
        """Fallback message when MRO has no match at all."""
        msg = get_user_message(NirspyError("test"))
        assert "unexpected error" in msg.lower()

    def test_mro_resolution_order(self) -> None:
        """More specific class wins over parent."""
        from nirspy.domain.exceptions import ValidationError
        from nirspy.engine.exceptions import MNEOperationError

        val_exc = ValidationError("test")
        mne_exc = MNEOperationError("test")

        val_msg = get_user_message(val_exc)
        mne_msg = get_user_message(mne_exc)

        assert val_msg != mne_msg
        assert val_msg == UI_ERROR_MESSAGES[ValidationError]
        assert mne_msg == UI_ERROR_MESSAGES[MNEOperationError]
