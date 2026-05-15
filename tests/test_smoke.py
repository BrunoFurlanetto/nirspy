"""Smoke test — garante que pacote importa."""

import nirspy


def test_version() -> None:
    assert nirspy.__version__
