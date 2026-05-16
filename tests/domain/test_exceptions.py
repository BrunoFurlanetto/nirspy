"""Tests for nirspy.domain.exceptions hierarchy."""

from __future__ import annotations

import pytest

from nirspy.domain.exceptions import DomainError, ExecutionError, ValidationError


class TestExceptionHierarchy:
    """Verify inheritance chain and basic instantiation."""

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
        with pytest.raises(Exception):
            raise ValidationError("oops")

    def test_multiple_errors_are_independent_instances(self) -> None:
        e1 = ValidationError("first")
        e2 = ValidationError("second")
        assert e1 is not e2
        assert str(e1) != str(e2)
