"""LoadSnirfBlock — loads a SNIRF file and returns a Raw MNE object.

Entry block for any fNIRS pipeline. Wraps :class:`~nirspy.engine.mne_adapter.MNEAdapter`
and produces a :class:`~nirspy.domain.data_types.DataType.RAW` result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne.io

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
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

    The block holds a stateless :class:`~nirspy.engine.mne_adapter.MNEAdapter` that
    is lazily constructed on first use so tests can swap it out.
    """

    def __init__(self, adapter: MNEAdapter | None = None) -> None:
        self._adapter = adapter or MNEAdapter()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _SPEC

    def run(self, data: Any, params: Any, context: Any) -> BlockResult:
        """Load the SNIRF file referenced in *params* and return a :class:`BlockResult`.

        Parameters
        ----------
        data:
            Ignored for the load block (no upstream data).
        params:
            A :class:`LoadSnirfParams` instance providing the file ``path``.
        context:
            :class:`~nirspy.domain.execution.ExecutionContext` — unused here but
            forwarded for interface compliance.

        Returns
        -------
        BlockResult
            ``data`` is the :class:`mne.io.BaseRaw` object.
            ``metadata`` contains ``n_channels`` and ``sfreq``.

        Raises
        ------
        ~nirspy.engine.exceptions.SnirfLoadError
            When the file is missing or cannot be parsed.
        """
        load_params: LoadSnirfParams = params
        raw: mne.io.BaseRaw = self._adapter.load_snirf(Path(load_params.path))

        return BlockResult(
            data=raw,
            block_id=_SPEC.block_id,
            metadata={
                "n_channels": len(raw.ch_names),
                "sfreq": raw.info["sfreq"],
            },
        )
