"""Shared fixtures for the nirspy test suite.

Domain tests are pure Python — no MNE, no disk I/O.
Engine tests may load real SNIRF data via MNE-NIRS sample dataset.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from nirspy.domain.block import Block, BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.execution import ExecutionContext
from nirspy.engine.cache_adapter import InMemoryCacheAdapter

# ---------------------------------------------------------------------------
# Fake Block implementation — used by domain tests (no MNE dependency)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeBlockImpl:
    """Minimal Block implementation that satisfies the Block Protocol.

    Implements the ADR-009 signature: ``run(context, inputs)``.
    Params are stored in ``self.params`` (set at construction time) and NOT
    passed through ``run``.  When ``inputs`` is non-empty, the single upstream
    value is passed through as-is so linear chain tests work correctly.
    """

    _spec: BlockSpec
    params: Any = None

    @property
    def spec(self) -> BlockSpec:
        return self._spec

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        # For a linear pipeline: if there is one upstream result, pass its data
        # through unchanged.  For the first block (inputs={}), return None so
        # tests can verify the empty-inputs contract explicitly.
        data = next(iter(inputs.values())) if inputs else None
        return BlockResult(data=data, block_id=self._spec.block_id)


def make_block(
    block_id: str,
    input_type: DataType,
    output_type: DataType,
    *,
    enabled: bool = True,
    params_class: type[Any] | None = None,
    description: str = "",
    params: Any = None,
) -> Block:
    """Factory that creates a lightweight fake Block for testing.

    Parameters
    ----------
    params:
        Optional params instance stored as ``block.params`` (ADR-009).
        Allows tests to verify that blocks access params via ``self``.
    """
    spec = BlockSpec(
        block_id=block_id,
        display_name=block_id,
        input_type=input_type,
        output_type=output_type,
        enabled=enabled,
        params_class=params_class,
        description=description,
    )
    return _FakeBlockImpl(spec, params=params)  # type: ignore[return-value]


@pytest.fixture()
def fake_block_raw_to_raw() -> Block:
    """A block that accepts RAW and produces RAW."""
    return make_block("raw_passthrough", DataType.RAW, DataType.RAW)


@pytest.fixture()
def fake_block_raw_to_od() -> Block:
    """A block that accepts RAW and produces RAW_OD."""
    return make_block("raw_to_od", DataType.RAW, DataType.RAW_OD)


@pytest.fixture()
def fake_block_od_to_haemo() -> Block:
    """A block that accepts RAW_OD and produces RAW_HAEMO."""
    return make_block("od_to_haemo", DataType.RAW_OD, DataType.RAW_HAEMO)


@pytest.fixture()
def in_memory_cache() -> InMemoryCacheAdapter:
    """Fresh InMemoryCacheAdapter per test."""
    return InMemoryCacheAdapter()


@pytest.fixture()
def default_context(in_memory_cache: InMemoryCacheAdapter) -> ExecutionContext:
    """ExecutionContext with in-memory cache and no-op progress."""
    return ExecutionContext(cache=in_memory_cache)


# ---------------------------------------------------------------------------
# SNIRF fixture (engine tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def snirf_path(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[return]
    """Return path to an MNE-NIRS sample SNIRF file.

    Downloads via mne.datasets if not already cached locally.
    Skips the test if no internet connection is available.
    """
    try:
        import mne
        import mne_nirs  # noqa: F401 — ensure package available

        data_dir = mne.datasets.fnirs_motor.data_path(verbose=False)
        import pathlib

        # MNE fnirs_motor dataset ships .snirf files
        snirf_files = list(pathlib.Path(data_dir).rglob("*.snirf"))
        if not snirf_files:
            pytest.skip("No SNIRF files found in fnirs_motor dataset.")
        return snirf_files[0]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not obtain SNIRF sample data: {exc}")
