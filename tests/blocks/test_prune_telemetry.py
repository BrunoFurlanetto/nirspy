"""Tests for PruneChannelsBlock telemetry (T-022).

Verifies that:
- BlockResult.metadata includes n_bads_total, n_channels_total, fraction_bads
- logging.warning emitted when fraction exceeds threshold
- Warning NOT emitted when fraction is below threshold
- Covers scenarios: 0%, 40%, 60%, 100% bads
"""

from __future__ import annotations

import logging

import mne
import numpy as np
import pytest

from nirspy.blocks.quality import PruneChannelsBlock, PruneChannelsParams
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.execution import ExecutionContext


@pytest.fixture()
def raw_od_6ch() -> mne.io.BaseRaw:
    """Raw OD with 6 channels (3 source-detector pairs)."""
    sfreq = 10.0
    n_times = int(10 * sfreq)
    ch_names = [
        "S1_D1 760", "S1_D1 850",
        "S2_D1 760", "S2_D1 850",
        "S3_D1 760", "S3_D1 850",
    ]
    ch_types = ["fnirs_od"] * 6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    sources = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.06, 0.0, 0.0]])
    detectors = np.array([[0.015, 0.0, 0.0]])
    for i, ch in enumerate(info["chs"]):
        src_idx = i // 2
        ch["loc"][3:6] = sources[src_idx]
        ch["loc"][6:9] = detectors[0]
        ch["loc"][9] = 760.0 if i % 2 == 0 else 850.0
    rng = np.random.default_rng(42)
    data = rng.standard_normal((6, n_times)) * 0.01
    return mne.io.RawArray(data, info, verbose=False)


def _make_context(sci_values: dict[str, float]) -> ExecutionContext:
    ctx = ExecutionContext()
    ctx.extra["sci_values"] = sci_values
    return ctx


class TestPruneTelemetryMetadata:
    """Metadata includes telemetry fields."""

    def test_zero_bads(self, raw_od_6ch: mne.io.BaseRaw) -> None:
        # All channels above threshold -> 0% bads
        sci = {ch: 0.9 for ch in raw_od_6ch.ch_names}
        ctx = _make_context(sci)
        block = PruneChannelsBlock()
        result = block.run(ctx, {"od": raw_od_6ch})

        assert result.metadata["n_bads_total"] == 0
        assert result.metadata["n_channels_total"] == 6
        assert result.metadata["fraction_bads"] == pytest.approx(0.0)

    def test_40_percent_bads(self, raw_od_6ch: mne.io.BaseRaw) -> None:
        # One pair below threshold -> 2/6 = 33% (pair expansion makes it 2)
        sci = {ch: 0.9 for ch in raw_od_6ch.ch_names}
        sci["S2_D1 760"] = 0.3
        sci["S2_D1 850"] = 0.3
        ctx = _make_context(sci)
        block = PruneChannelsBlock()
        result = block.run(ctx, {"od": raw_od_6ch})

        assert result.metadata["n_bads_total"] == 2
        assert result.metadata["n_channels_total"] == 6
        assert result.metadata["fraction_bads"] == pytest.approx(2 / 6)

    def test_60_percent_bads(self, raw_od_6ch: mne.io.BaseRaw) -> None:
        # Two pairs below threshold -> 4/6 = 67%
        sci = {ch: 0.9 for ch in raw_od_6ch.ch_names}
        sci["S2_D1 760"] = 0.3
        sci["S2_D1 850"] = 0.3
        sci["S3_D1 760"] = 0.2
        sci["S3_D1 850"] = 0.2
        ctx = _make_context(sci)
        block = PruneChannelsBlock()
        result = block.run(ctx, {"od": raw_od_6ch})

        assert result.metadata["n_bads_total"] == 4
        assert result.metadata["n_channels_total"] == 6
        assert result.metadata["fraction_bads"] == pytest.approx(4 / 6)

    def test_100_percent_bads_raises(self, raw_od_6ch: mne.io.BaseRaw) -> None:
        # All channels below threshold -> ValidationError (existing guard)
        sci = {ch: 0.1 for ch in raw_od_6ch.ch_names}
        ctx = _make_context(sci)
        block = PruneChannelsBlock()

        with pytest.raises(ValidationError, match="every channel"):
            block.run(ctx, {"od": raw_od_6ch})


class TestPruneTelemetryWarning:
    """Warning emitted above threshold, silent below."""

    def test_warning_above_threshold(
        self,
        raw_od_6ch: mne.io.BaseRaw,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # 4/6 = 67% > 50% default threshold
        sci = {ch: 0.9 for ch in raw_od_6ch.ch_names}
        sci["S2_D1 760"] = 0.3
        sci["S2_D1 850"] = 0.3
        sci["S3_D1 760"] = 0.2
        sci["S3_D1 850"] = 0.2
        ctx = _make_context(sci)
        block = PruneChannelsBlock()

        with caplog.at_level(logging.WARNING, logger="nirspy.blocks.quality"):
            block.run(ctx, {"od": raw_od_6ch})

        assert any("marked as bad" in msg for msg in caplog.messages)

    def test_no_warning_below_threshold(
        self,
        raw_od_6ch: mne.io.BaseRaw,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # 2/6 = 33% < 50% default threshold
        sci = {ch: 0.9 for ch in raw_od_6ch.ch_names}
        sci["S2_D1 760"] = 0.3
        sci["S2_D1 850"] = 0.3
        ctx = _make_context(sci)
        block = PruneChannelsBlock()

        with caplog.at_level(logging.WARNING, logger="nirspy.blocks.quality"):
            block.run(ctx, {"od": raw_od_6ch})

        assert not any("marked as bad" in msg for msg in caplog.messages)

    def test_custom_warning_threshold(
        self,
        raw_od_6ch: mne.io.BaseRaw,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # 2/6 = 33% > 25% custom threshold -> warning
        sci = {ch: 0.9 for ch in raw_od_6ch.ch_names}
        sci["S2_D1 760"] = 0.3
        sci["S2_D1 850"] = 0.3
        ctx = _make_context(sci)
        block = PruneChannelsBlock(
            params=PruneChannelsParams(bad_fraction_warning=0.25)
        )

        with caplog.at_level(logging.WARNING, logger="nirspy.blocks.quality"):
            block.run(ctx, {"od": raw_od_6ch})

        assert any("marked as bad" in msg for msg in caplog.messages)

    def test_zero_bads_no_warning(
        self,
        raw_od_6ch: mne.io.BaseRaw,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sci = {ch: 0.9 for ch in raw_od_6ch.ch_names}
        ctx = _make_context(sci)
        block = PruneChannelsBlock()

        with caplog.at_level(logging.WARNING, logger="nirspy.blocks.quality"):
            block.run(ctx, {"od": raw_od_6ch})

        assert not any("marked as bad" in msg for msg in caplog.messages)
