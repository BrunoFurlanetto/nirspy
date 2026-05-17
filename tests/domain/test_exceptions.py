"""Tests for nirspy.domain.exceptions hierarchy.

NirspyError is the package-wide root.
DomainError and EngineError both inherit from NirspyError so that GUI
code can use ``except NirspyError`` to catch all package errors without
coupling to specific subclasses.
"""

from __future__ import annotations

import pytest

from nirspy.domain.exceptions import DomainError, ExecutionError, NirspyError, ValidationError
from nirspy.engine.exceptions import EngineError, MNEOperationError, SnirfLoadError


class TestNirspyErrorRoot:
    """NirspyError is the top of the package hierarchy."""

    def test_nirspy_error_is_exception(self) -> None:
        assert issubclass(NirspyError, Exception)

    def test_domain_error_inherits_nirspy_error(self) -> None:
        assert issubclass(DomainError, NirspyError)

    def test_engine_error_inherits_nirspy_error(self) -> None:
        assert issubclass(EngineError, NirspyError)

    def test_gui_can_catch_domain_error_as_nirspy_error(self) -> None:
        with pytest.raises(NirspyError):
            raise DomainError("domain problem")

    def test_gui_can_catch_engine_error_as_nirspy_error(self) -> None:
        with pytest.raises(NirspyError):
            raise EngineError("engine problem")

    def test_gui_can_catch_validation_error_as_nirspy_error(self) -> None:
        with pytest.raises(NirspyError):
            raise ValidationError("bad config")

    def test_gui_can_catch_execution_error_as_nirspy_error(self) -> None:
        with pytest.raises(NirspyError):
            raise ExecutionError("block failed")

    def test_gui_can_catch_mne_operation_error_as_nirspy_error(self) -> None:
        with pytest.raises(NirspyError):
            raise MNEOperationError("mne failed")

    def test_gui_can_catch_snirf_load_error_as_nirspy_error(self) -> None:
        with pytest.raises(NirspyError):
            raise SnirfLoadError("bad snirf")

    def test_nirspy_error_does_not_hide_stdlib_exceptions(self) -> None:
        """A plain ValueError must NOT be caught by except NirspyError."""
        with pytest.raises(ValueError):
            try:
                raise ValueError("stdlib error")
            except NirspyError:
                pass  # must NOT reach here


class TestDomainExceptionHierarchy:
    """DomainError subtree invariants."""

    def test_domain_error_is_exception(self) -> None:
        assert issubclass(DomainError, Exception)

    def test_validation_error_inherits_domain_error(self) -> None:
        assert issubclass(ValidationError, DomainError)

    def test_execution_error_inherits_domain_error(self) -> None:
        assert issubclass(ExecutionError, DomainError)

    def test_can_catch_validation_as_domain_error(self) -> None:
        with pytest.raises(DomainError):
            raise ValidationError("bad config")

    def test_can_catch_execution_as_domain_error(self) -> None:
        with pytest.raises(DomainError):
            raise ExecutionError("block failed")

    def test_validation_error_carries_message(self) -> None:
        exc = ValidationError("type mismatch at step 2")
        assert "type mismatch" in str(exc)

    def test_execution_error_carries_message(self) -> None:
        exc = ExecutionError("block 'od_1' crashed")
        assert "od_1" in str(exc)

    def test_validation_error_is_catchable_as_base_exception(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 - intentional: verify Exception subclass
            raise ValidationError("oops")

    def test_multiple_errors_are_independent_instances(self) -> None:
        e1 = ValidationError("first")
        e2 = ValidationError("second")
        assert e1 is not e2
        assert str(e1) != str(e2)


class TestEngineExceptionHierarchy:
    """EngineError subtree — lives in engine layer but root is NirspyError."""

    def test_engine_error_is_not_domain_error(self) -> None:
        # EngineError does NOT inherit DomainError — they are siblings under NirspyError.
        assert not issubclass(EngineError, DomainError)

    def test_mne_operation_error_inherits_engine_error(self) -> None:
        assert issubclass(MNEOperationError, EngineError)

    def test_snirf_load_error_inherits_mne_operation_error(self) -> None:
        assert issubclass(SnirfLoadError, MNEOperationError)

    def test_mne_operation_error_stores_original_exception(self) -> None:
        original = RuntimeError("mne internal failure")
        exc = MNEOperationError("adapter failed", mne_exception=original)
        assert exc.mne_exception is original

    def test_mne_operation_error_without_original_exception(self) -> None:
        exc = MNEOperationError("adapter failed")
        assert exc.mne_exception is None

    def test_snirf_load_error_carries_message(self) -> None:
        exc = SnirfLoadError("file.snirf not found")
        assert "file.snirf" in str(exc)

    def test_can_catch_snirf_load_error_as_mne_operation_error(self) -> None:
        with pytest.raises(MNEOperationError):
            raise SnirfLoadError("bad snirf")

    def test_can_catch_snirf_load_error_as_engine_error(self) -> None:
        with pytest.raises(EngineError):
            raise SnirfLoadError("bad snirf")
