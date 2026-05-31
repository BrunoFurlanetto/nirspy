"""DataType enum — describes the kind of data flowing between blocks."""

from __future__ import annotations

from enum import Enum


class DataType(str, Enum):
    """Semantic type of the data object passed between pipeline blocks.

    Using ``str`` mixin allows JSON/YAML serialization without a custom encoder.
    """

    NONE = "none"
    """No data — used for source blocks (no input) or sink blocks (no output).

    A block with ``input_type=NONE`` must be the first step in the pipeline
    (it reads from disk, not from an upstream block).  A block with
    ``output_type=NONE`` must be the last step (it writes to disk or emits a
    side-effect).  See ADR-019.
    """

    RAW = "raw"
    """MNE Raw object (intensity or OD)."""

    RAW_OD = "raw_od"
    """Optical density Raw."""

    RAW_HAEMO = "raw_haemo"
    """Haemodynamic (HbO/HbR) Raw after Beer-Lambert."""

    EPOCHS = "epochs"
    """MNE Epochs."""

    EVOKED = "evoked"
    """MNE Evoked."""

    GLM_RESULT = "glm_result"
    """GLM statistical results (coefficients, t-stats, p-values per channel)."""

    DATAFRAME = "dataframe"
    """Generic pandas DataFrame."""

    ANY = "any"
    """Wildcard — block accepts or produces any type (use sparingly)."""
