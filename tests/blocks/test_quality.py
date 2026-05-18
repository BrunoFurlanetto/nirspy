"""Tests for quality control blocks (T-004)."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspy.blocks.quality import (
    PruneChannelsBlock,
    PruneChannelsParams,
    ScalpCouplingIndexBlock,
    ScalpCouplingIndexParams,
)
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.execution import ExecutionContext


@pytest.fixture()
def raw_od_with_cardiac() -> mne.io.BaseRaw:
    """Raw OD data with cardiac component for SCI."""
    sfreq = 10.0
    n_times = int(30 * sfreq)
    n_channels = 4
    ch_names = ["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"]
    ch_types = ["fnirs_od"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    sources = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
    detectors = np.array([[0.015, 0.0, 0.0]])
    for i, ch in enumerate(info["chs"]):
        src_idx = i // 2
        ch["loc"][3:6] = sources[src_idx]
        ch["loc"][6:9] = detectors[0]
        wavelength = 760.0 if i % 2 == 0 else 850.0
        ch["loc"][9] = wavelength
    rng = np.random.default_rng(42)
    t = np.arange(n_times) / sfreq
    data = np.zeros((n_channels, n_times))
    for i in range(n_channels):
        if i < 2:
            cardiac = 0.01 * np.sin(2 * np.pi * 1.0 * t)
            noise = rng.normal(0, 0.001, n_times)
            data[i] = cardiac + noise
        else:
            data[i] = rng.normal(0, 0.01, n_times)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture()
def context() -> ExecutionContext:
    """Fresh execution context."""
    return ExecutionContext()


@pytest.fixture()
def context_with_sci() -> ExecutionContext:
    """Context with SCI values."""
    ctx = ExecutionContext()
    ctx.extra["sci_values"] = {
        "S1_D1 760": 0.9, "S1_D1 850": 0.85,
        "S2_D1 760": 0.3, "S2_D1 850": 0.2,
    }
    return ctx


class TestScalpCouplingIndexBlock:
    """Tests for ScalpCouplingIndexBlock."""

    def test_returns_sci_in_metadata(self, raw_od_with_cardiac, context):
        block = ScalpCouplingIndexBlock()
        result = block.run(context, {"optical_density": raw_od_with_cardiac})
        assert "sci_values" in result.metadata
        assert isinstance(result.metadata["sci_values"], dict)
        assert set(result.metadata["sci_values"].keys()) == set(raw_od_with_cardiac.ch_names)

    def test_sci_values_in_range(self, raw_od_with_cardiac, context):
        block = ScalpCouplingIndexBlock()
        result = block.run(context, {"optical_density": raw_od_with_cardiac})
        for val in result.metadata["sci_values"].values():
            assert -1.0 <= val <= 1.0  # SCI can be negative for poor coupling

    def test_output_is_same_raw(self, raw_od_with_cardiac, context):
        block = ScalpCouplingIndexBlock()
        result = block.run(context, {"optical_density": raw_od_with_cardiac})
        assert result.data is raw_od_with_cardiac

    def test_block_id(self, raw_od_with_cardiac, context):
        block = ScalpCouplingIndexBlock()
        result = block.run(context, {"optical_density": raw_od_with_cardiac})
        assert result.block_id == "scalp_coupling_index"

    def test_raises_on_empty_inputs(self, context):
        block = ScalpCouplingIndexBlock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(context, {})

    def test_raises_on_wrong_ch_type(self, context):
        info = mne.create_info(
            ch_names=["S1_D1 760", "S1_D1 850"], sfreq=10.0,
            ch_types=["fnirs_cw_amplitude", "fnirs_cw_amplitude"],
        )
        for i, ch in enumerate(info["chs"]):
            ch["loc"][3:6] = [0.0, 0.0, 0.0]
            ch["loc"][6:9] = [0.015, 0.0, 0.0]
            ch["loc"][9] = 760.0 if i == 0 else 850.0
        data = np.ones((2, 100))
        raw = mne.io.RawArray(data, info, verbose=False)
        block = ScalpCouplingIndexBlock()
        with pytest.raises(ValidationError, match="fnirs_od"):
            block.run(context, {"load_snirf": raw})

    def test_spec(self):
        block = ScalpCouplingIndexBlock()
        assert block.spec.block_id == "scalp_coupling_index"
        assert block.spec.params_class is ScalpCouplingIndexParams


class TestPruneChannelsBlock:
    """Tests for PruneChannelsBlock."""

    def test_marks_low_sci_as_bads(self, raw_od_with_cardiac, context_with_sci):
        block = PruneChannelsBlock(PruneChannelsParams(sci_threshold=0.5))
        result = block.run(context_with_sci, {"scalp_coupling_index": raw_od_with_cardiac})
        assert "S2_D1 760" in result.data.info["bads"]
        assert "S2_D1 850" in result.data.info["bads"]

    def test_does_not_mark_good(self, raw_od_with_cardiac, context_with_sci):
        block = PruneChannelsBlock(PruneChannelsParams(sci_threshold=0.5))
        result = block.run(context_with_sci, {"scalp_coupling_index": raw_od_with_cardiac})
        assert "S1_D1 760" not in result.data.info["bads"]
        assert "S1_D1 850" not in result.data.info["bads"]

    def test_does_not_remove_channels(self, raw_od_with_cardiac, context_with_sci):
        block = PruneChannelsBlock(PruneChannelsParams(sci_threshold=0.5))
        result = block.run(context_with_sci, {"scalp_coupling_index": raw_od_with_cardiac})
        assert len(result.data.ch_names) == len(raw_od_with_cardiac.ch_names)

    def test_metadata_has_pruned_info(self, raw_od_with_cardiac, context_with_sci):
        block = PruneChannelsBlock(PruneChannelsParams(sci_threshold=0.5))
        result = block.run(context_with_sci, {"scalp_coupling_index": raw_od_with_cardiac})
        assert "pruned_channels" in result.metadata
        assert result.metadata["n_pruned"] == 2

    def test_raises_without_sci(self, raw_od_with_cardiac, context):
        block = PruneChannelsBlock()
        with pytest.raises(ValidationError, match="SCI values"):
            block.run(context, {"scalp_coupling_index": raw_od_with_cardiac})

    def test_raises_on_empty_inputs(self, context_with_sci):
        block = PruneChannelsBlock()
        with pytest.raises(ValidationError, match="requires input data"):
            block.run(context_with_sci, {})

    def test_raises_on_bad_threshold(self, raw_od_with_cardiac, context_with_sci):
        block = PruneChannelsBlock(PruneChannelsParams(sci_threshold=1.5))
        with pytest.raises(ValidationError, match="sci_threshold must be in"):
            block.run(context_with_sci, {"scalp_coupling_index": raw_od_with_cardiac})

    def test_preserves_existing_bads(self, raw_od_with_cardiac, context_with_sci):
        raw_od_with_cardiac.info["bads"] = ["S1_D1 760"]
        block = PruneChannelsBlock(PruneChannelsParams(sci_threshold=0.5))
        result = block.run(context_with_sci, {"scalp_coupling_index": raw_od_with_cardiac})
        assert "S1_D1 760" in result.data.info["bads"]
        assert "S2_D1 760" in result.data.info["bads"]

    def test_threshold_zero(self, raw_od_with_cardiac, context_with_sci):
        block = PruneChannelsBlock(PruneChannelsParams(sci_threshold=0.0))
        result = block.run(context_with_sci, {"scalp_coupling_index": raw_od_with_cardiac})
        assert result.metadata["n_pruned"] == 0

    def test_threshold_one_marks_all_raises(self, raw_od_with_cardiac, context_with_sci):
        block = PruneChannelsBlock(PruneChannelsParams(sci_threshold=1.0))
        with pytest.raises(ValidationError, match="every channel"):
            block.run(context_with_sci, {"scalp_coupling_index": raw_od_with_cardiac})

    def test_spec(self):
        block = PruneChannelsBlock()
        assert block.spec.block_id == "prune_channels"
        assert block.spec.params_class is PruneChannelsParams

    def test_output_is_copy(self, raw_od_with_cardiac, context_with_sci):
        block = PruneChannelsBlock(PruneChannelsParams(sci_threshold=0.5))
        result = block.run(context_with_sci, {"scalp_coupling_index": raw_od_with_cardiac})
        assert result.data is not raw_od_with_cardiac
        assert raw_od_with_cardiac.info["bads"] == []
