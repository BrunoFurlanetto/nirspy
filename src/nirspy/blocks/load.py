"""LoadSnirfBlock — loads a SNIRF file and returns a Raw MNE object.

Entry block for any fNIRS pipeline. Wraps :class:`~nirspy.engine.mne_adapter.MNEAdapter`
and produces a :class:`~nirspy.domain.data_types.DataType.RAW` result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import mne.io

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ExecutionError
from nirspy.engine.exceptions import MNEOperationError, SnirfLoadError
from nirspy.engine.mne_adapter import MNEAdapter


@dataclass(frozen=True)
class LoadSnirfParams:
    """Parameters for :class:`LoadSnirfBlock`.

    Attributes
    ----------
    path:
        Absolute or relative path to the ``.snirf`` file to load.
    """

    path: str


_SPEC = BlockSpec(
    block_id="load_snirf",
    display_name="Load SNIRF",
    description="Load a SNIRF file and emit a MNE Raw object.",
    input_type=DataType.ANY,
    output_type=DataType.RAW,
    params_class=LoadSnirfParams,
)


class LoadSnirfBlock:
    """Block that loads a SNIRF file via MNE-NIRS and emits ``DataType.RAW``.

    Parameters
    ----------
    params:
        :class:`LoadSnirfParams` instance holding the file path. Stored as
        ``self.params`` and accessed in :meth:`run` (ADR-009).
    adapter:
        Optional :class:`~nirspy.engine.mne_adapter.MNEAdapter` override.
        Defaults to a freshly constructed :class:`MNEAdapter` so that tests
        can inject a fake adapter without modifying production code.

    Class attributes
    ----------------
    SPEC:
        Class-level reference to the static :class:`~nirspy.domain.block.BlockSpec`
        descriptor. Exposed here so the serialiser can read ``params_class``
        from the class without instantiating it.
    """

    SPEC: ClassVar[BlockSpec] = _SPEC

    def __init__(
        self,
        params: LoadSnirfParams,
        adapter: MNEAdapter | None = None,
    ) -> None:
        self.params: LoadSnirfParams = params
        self._adapter: MNEAdapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Load the SNIRF file referenced in ``self.params`` and return a :class:`BlockResult`.

        Parameters
        ----------
        context:
            :class:`~nirspy.domain.execution.ExecutionContext` injected by the
            runner — unused here but required by the :class:`~nirspy.domain.block.Block`
            Protocol.
        inputs:
            Must be an empty dict (``{}``).  The load block is always the first
            step in a linear pipeline and has no upstream dependency.

        Returns
        -------
        BlockResult
            ``data`` is the :class:`mne.io.BaseRaw` object.
            ``metadata`` contains ``n_channels`` and ``sfreq``.

        Raises
        ------
        ExecutionError
            When *inputs* is non-empty (contract violation: load block must be first).
        ~nirspy.engine.exceptions.SnirfLoadError
            When the file is missing or cannot be parsed (propagated from adapter).
        ~nirspy.engine.exceptions.MNEOperationError
            When MNE raises an unexpected error during loading.
        """
        if inputs:
            raise ExecutionError(
                "LoadSnirfBlock received non-empty inputs. "
                "The load block must be the first block in the pipeline (inputs must be {})."
            )

        try:
            raw: mne.io.BaseRaw = self._adapter.load_snirf(Path(self.params.path))
        except (SnirfLoadError, MNEOperationError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(
                f"LoadSnirfBlock failed to load '{self.params.path}': {exc}"
            ) from exc

        return BlockResult(
            data=raw,
            block_id=_SPEC.block_id,
            metadata={
                "n_channels": len(raw.ch_names),
                "sfreq": raw.info["sfreq"],
            },
        )
