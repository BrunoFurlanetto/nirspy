"""Tests for ShortChannelRegressionBlock (T-033)."""

from __future__ import annotations

from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from nirspy.blocks.signal_enhancement import (
    ShortChannelRegressionBlock,
    ShortChannelRegressionParams,
)
from nirspy.domain.block import BlockResult
from nirspy.domain.exceptions import ValidationError


@pytest.fixture()
def raw_haemo_with_short_channels() -> mne.io.BaseRaw:
    """Create synthetic Raw with hbo/hbr channels including short channels.

    4 long channels (2 S-D pairs x 2 chromophores) at 30mm distance
    + 2 short channels (1 S-D pair x 2 chromophores) at 8mm distance.
    """
    sfreq = 10.0
    n_times = int(10 * sfreq)

    # Long channels: S1-D1 (30mm apart)
    # Short channels: S1-D2 (8mm apart)
    ch_names = [
        "S1_D1 hbo",
        "S1_D1 hbr",
        "S2_D1 hbo",
        "S2_D1 hbr",
        "S1_D2 hbo",
        "S1_D2 hbr",
    ]
    ch_types = ["hbo", "hbr", "hbo", "hbr", "hbo", "hbr"]
    n_channels = len(ch_names)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Set source/detector positions
    sources = np.array([
        [0.0, 0.0, 0.0],   # S1
        [0.05, 0.0, 0.0],  # S2
    ])
    detectors = np.array([
        [0.03, 0.0, 0.0],  # D1 (30mm from S1, 20mm from S2)
        [0.008, 0.0, 0.0],  # D2 (8mm from S1 -- short channel)
    ])

    # Assign locations
    # S1_D1 hbo, S1_D1 hbr
    for i in [0, 1]:
        info["chs"][i]["loc"][3:6] = sources[0]
        info["chs"][i]["loc"][6:9] = detectors[0]
    # S2_D1 hbo, S2_D1 hbr
    for i in [2, 3]:
        info["chs"][i]["loc"][3:6] = sources[1]
        info["chs"][i]["loc"][6:9] = detectors[0]
    # S1_D2 hbo, S1_D2 hbr (short)
    for i in [4, 5]:
        info["chs"][i]["loc"][3:6] = sources[0]
        info["chs"][i]["loc"][6:9] = detectors[1]

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 1e-6

    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


class TestShortChannelRegressionBlock:
    """Tests for ShortChannelRegressionBlock."""

    def test_params_defaults(self) -> None:
        """Default params have max_dist=0.015."""
        params = ShortChannelRegressionParams()
        assert params.max_dist == 0.015

    def test_params_frozen(self) -> None:
        """Params are immutable."""
        params = ShortChannelRegressionParams()
        with pytest.raises(AttributeError):
            params.max_dist = 0.02  # type: ignore[misc]

    def test_spec_correct(self) -> None:
        """Block spec has correct input/output types."""
        from nirspy.domain.data_types import DataType

        block = ShortChannelRegressionBlock()
        assert block.spec.block_id == "short_channel_regression"
        assert block.spec.input_type == DataType.RAW_HAEMO
        assert block.spec.output_type == DataType.RAW_HAEMO

    def test_raises_on_empty_inputs(self) -> None:
        """Block raises ValidationError when inputs is empty."""
        block = ShortChannelRegressionBlock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(None, {})

    def test_raises_on_wrong_channel_type(self) -> None:
        """Block raises ValidationError when channels are not hbo/hbr."""
        block = ShortChannelRegressionBlock()
        # Create raw with wrong channel type
        info = mne.create_info(
            ch_names=["S1_D1 760", "S1_D1 850"],
            sfreq=10.0,
            ch_types=["fnirs_cw_amplitude", "fnirs_cw_amplitude"],
        )
        for i, ch in enumerate(info["chs"]):
            ch["loc"][3:6] = [0.0, 0.0, 0.0]
            ch["loc"][6:9] = [0.03, 0.0, 0.0]
            ch["loc"][9] = 760.0 if i == 0 else 850.0

        raw = mne.io.RawArray(
            np.ones((2, 100)), info, verbose=False
        )
        with pytest.raises(ValidationError, match="expects hbo/hbr"):
            block.run(None, {"prev": raw})

    def test_raises_on_invalid_max_dist(
        self, raw_haemo_with_short_channels: mne.io.BaseRaw
    ) -> None:
        """Block raises ValidationError when max_dist <= 0."""
        params = ShortChannelRegressionParams(max_dist=-0.01)
        block = ShortChannelRegressionBlock(params=params)
        with pytest.raises(ValidationError, match="max_dist must be > 0"):
            block.run(None, {"prev": raw_haemo_with_short_channels})

    def test_run_calls_adapter(
        self, raw_haemo_with_short_channels: mne.io.BaseRaw
    ) -> None:
        """Block delegates to MNEAdapter.short_channel_regression."""
        adapter = MagicMock(spec=["short_channel_regression"])
        adapter.short_channel_regression.return_value = (
            raw_haemo_with_short_channels
        )

        params = ShortChannelRegressionParams(max_dist=0.015)
        block = ShortChannelRegressionBlock(params=params, adapter=adapter)
        result = block.run(None, {"prev": raw_haemo_with_short_channels})

        adapter.short_channel_regression.assert_called_once_with(
            raw_haemo_with_short_channels, max_dist=0.015
        )
        assert isinstance(result, BlockResult)
        assert result.block_id == "short_channel_regression"

    def test_metadata_contains_channel_info(
        self, raw_haemo_with_short_channels: mne.io.BaseRaw
    ) -> None:
        """Metadata reports short/long channel counts."""
        adapter = MagicMock(spec=["short_channel_regression"])
        adapter.short_channel_regression.return_value = (
            raw_haemo_with_short_channels
        )

        block = ShortChannelRegressionBlock(
            params=ShortChannelRegressionParams(max_dist=0.015),
            adapter=adapter,
        )
        result = block.run(None, {"prev": raw_haemo_with_short_channels})

        assert result.metadata["max_dist"] == 0.015
        assert result.metadata["n_short_channels"] == 2  # S1_D2 hbo + hbr
        assert result.metadata["n_long_channels"] == 4

    def test_registered_in_registry(self) -> None:
        """Block is registered in the global registry."""
        from nirspy.blocks import registry

        assert "short_channel_regression" in registry.list_blocks()
