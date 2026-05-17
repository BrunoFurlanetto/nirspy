"""LoadSnirfBlock — loads a SNIRF file and returns a Raw MNE object.

Entry block for any fNIRS pipeline. Wraps :class:`~nirspy.engine.mne_adapter.MNEAdapter`
and produces a :class:`~nirspy.domain.data_types.DataType.RAW` result.

Security (S-02): Path is canonicalized via ``resolve()`` and validated against
an allowlist of directories. Paths containing ``..`` components are rejected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import mne.io

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ExecutionError, ValidationError
from nirspy.engine.exceptions import MNEOperationError, SnirfLoadError
from nirspy.engine.mne_adapter import MNEAdapter

# ---------------------------------------------------------------------------
# Security: default allowed directories for file loading (S-02)
# ---------------------------------------------------------------------------

_DEFAULT_ALLOWED_DIRS: list[Path] = [
    Path.home(),
    Path.cwd(),
]


def get_allowed_dirs() -> list[Path]:
    """Return the list of allowed base directories for file loading.

    Directories are resolved (absolute, no symlinks). Defaults to user home
    and current working directory. Can be extended via ``NIRSPY_ALLOWED_DIRS``
    environment variable (``os.pathsep``-separated paths).
    """
    dirs = [p.resolve() for p in _DEFAULT_ALLOWED_DIRS]
    env_dirs = os.environ.get("NIRSPY_ALLOWED_DIRS", "")
    if env_dirs:
        for d in env_dirs.split(os.pathsep):
            p = Path(d).resolve()
            if p.is_dir():
                dirs.append(p)
    return dirs


def validate_snirf_path(path: Path, allowed_dirs: list[Path] | None = None) -> Path:
    """Validate and canonicalize a SNIRF file path (S-02).

    Parameters
    ----------
    path:
        User-provided path to validate.
    allowed_dirs:
        Optional override for testing. Defaults to :func:`get_allowed_dirs`.

    Returns
    -------
    Path
        Resolved (canonical) path guaranteed to be within allowed directories.

    Raises
    ------
    ValidationError
        When the path contains ``..`` traversal, is outside allowed directories,
        or does not have a ``.snirf`` extension.
    """
    # Reject raw path containing ".." before resolution
    path_str = str(path)
    if ".." in path_str.replace("\\", "/").split("/"):
        raise ValidationError(
            f"Path traversal detected: '{path}'. "
            "Paths containing '..' components are not allowed."
        )

    resolved = path.resolve()

    # Validate extension
    if resolved.suffix.lower() != ".snirf":
        raise ValidationError(
            f"Expected a .snirf file, got '{resolved.suffix}': {resolved}"
        )

    # Validate against allowlist
    dirs = allowed_dirs if allowed_dirs is not None else get_allowed_dirs()
    for allowed in dirs:
        try:
            resolved.relative_to(allowed)
            return resolved
        except ValueError:
            continue

    raise ValidationError(
        f"Path '{resolved}' is outside allowed directories. "
        f"Allowed: {[str(d) for d in dirs]}"
    )


# ---------------------------------------------------------------------------
# Block definition
# ---------------------------------------------------------------------------


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
    allowed_dirs:
        Optional override for path validation allowlist.  When *None*
        (default), uses :func:`get_allowed_dirs`.

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
        allowed_dirs: list[Path] | None = None,
    ) -> None:
        self.params: LoadSnirfParams = params
        self._adapter: MNEAdapter = adapter or MNEAdapter()
        self._allowed_dirs = allowed_dirs

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Load the SNIRF file referenced in ``self.params`` and return a :class:`BlockResult`.

        Security (S-02): The path is validated and canonicalized before loading.

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
        ValidationError
            When the path fails security validation (traversal, outside allowlist).
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

        # S-02: Validate path before loading
        validated_path = validate_snirf_path(
            Path(self.params.path), self._allowed_dirs
        )

        try:
            raw: mne.io.BaseRaw = self._adapter.load_snirf(validated_path)
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
