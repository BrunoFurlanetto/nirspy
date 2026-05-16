"""DataType enum — describes the kind of data flowing between blocks."""

from __future__ import annotations

from enum import Enum


class DataType(str, Enum):
    """Semantic type of the data object passed between pipeline blocks.

    Using ``str`` mixin allows JSON/YAML serialization without a custom encoder.
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

    DATAFRAME = "dataframe"
    """Generic pandas DataFrame."""

    ANY = "any"
    """Wildcard — block accepts or produces any type (use sparingly)."""
