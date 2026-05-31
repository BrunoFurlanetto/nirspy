"""Tests for GLMBlock -- T-034."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspy.blocks.glm import GLMBlock, GLMParams
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.glm_result import GLMResult


@pytest.fixture()
def raw_haemo_with_events(raw_haemo: mne.io.BaseRaw) -> mne.io.BaseRaw:
    sfreq = 10.0
    n_times = int(60 * sfreq)
    ch_names = ["S1_D1 hbo", "S1_D1 hbr"]
    ch_types = ["hbo", "hbr"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    for ch in info["chs"]:
        ch["loc"][3:6] = np.array([0.0, 0.0, 0.0])
        ch["loc"][6:9] = np.array([0.03, 0.0, 0.0])
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1e-6, size=(2, n_times))
    raw = mne.io.RawArray(data, info, verbose=False)
    ann = mne.Annotations(onset=[5.0, 20.0, 40.0], duration=[1.0]*3, description=["stimulus_A"]*3)
    raw.set_annotations(ann)
    return raw


@pytest.fixture()
def raw_haemo_multi_condition() -> mne.io.BaseRaw:
    sfreq = 10.0
    n_times = int(90 * sfreq)
    ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"]
    ch_types = ["hbo", "hbr", "hbo", "hbr"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    for ch in info["chs"]:
        ch["loc"][3:6] = np.array([0.0, 0.0, 0.0])
        ch["loc"][6:9] = np.array([0.03, 0.0, 0.0])
    rng = np.random.default_rng(123)
    data = rng.normal(0, 1e-6, size=(4, n_times))
    raw = mne.io.RawArray(data, info, verbose=False)
    descs = ["cond_A", "cond_B", "cond_A", "cond_B", "cond_A", "cond_B"]
    ann = mne.Annotations(onset=[5.0,15.0,30.0,45.0,60.0,75.0], duration=[1.0]*6, description=descs)
    raw.set_annotations(ann)
    return raw


class TestGLMBlock:
    def test_spec_types(self) -> None:
        block = GLMBlock()
        assert block.spec.input_type == DataType.RAW_HAEMO
        assert block.spec.output_type == DataType.GLM_RESULT
        assert block.spec.block_id == "glm"

    def test_run_produces_glm_result(self, raw_haemo_with_events) -> None:
        result = GLMBlock().run(None, {"upstream": raw_haemo_with_events})
        assert result.block_id == "glm"
        assert isinstance(result.data, GLMResult)

    def test_glm_result_attributes(self, raw_haemo_with_events) -> None:
        result = GLMBlock().run(None, {"upstream": raw_haemo_with_events})
        glm = result.data
        assert len(glm.channel_names) == 2
        assert "stimulus_A" in glm.regressor_names
        n_reg = len(glm.regressor_names)
        n_ch = len(glm.channel_names)
        assert glm.theta.shape == (n_reg, n_ch)
        assert glm.t_stats.shape == (n_reg, n_ch)
        assert glm.p_values.shape == (n_reg, n_ch)
        assert glm.mse.shape == (n_ch,)

    def test_to_dataframe(self, raw_haemo_with_events) -> None:
        import pandas as pd
        result = GLMBlock().run(None, {"upstream": raw_haemo_with_events})
        df = result.data.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "regressor" in df.columns

    def test_get_contrast(self, raw_haemo_with_events) -> None:
        result = GLMBlock().run(None, {"upstream": raw_haemo_with_events})
        t_s = result.data.get_contrast("stimulus_A")
        assert t_s.shape == (len(result.data.channel_names),)
        assert np.all(np.isfinite(t_s))

    def test_get_contrast_invalid(self, raw_haemo_with_events) -> None:
        result = GLMBlock().run(None, {"upstream": raw_haemo_with_events})
        with pytest.raises(KeyError):
            result.data.get_contrast("nonexistent")

    def test_multi_condition(self, raw_haemo_multi_condition) -> None:
        result = GLMBlock().run(None, {"upstream": raw_haemo_multi_condition})
        assert "cond_A" in result.data.regressor_names
        assert "cond_B" in result.data.regressor_names

    def test_no_inputs_raises(self) -> None:
        with pytest.raises(ValidationError, match="requires input"):
            GLMBlock().run(None, {})

    def test_wrong_channel_type_raises(self) -> None:
        info = mne.create_info(["ch1"], sfreq=10.0, ch_types=["eeg"])
        raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose=False)
        with pytest.raises(ValidationError, match="hbo/hbr"):
            GLMBlock().run(None, {"upstream": raw})

    def test_invalid_drift_model(self, raw_haemo_with_events) -> None:
        with pytest.raises(ValidationError, match="drift_model"):
            GLMBlock(params=GLMParams(drift_model="bad")).run(
                None, {"upstream": raw_haemo_with_events},
            )

    def test_invalid_hrf_model(self, raw_haemo_with_events) -> None:
        with pytest.raises(ValidationError, match="hrf_model"):
            GLMBlock(params=GLMParams(hrf_model="bad")).run(
                None, {"upstream": raw_haemo_with_events},
            )

    def test_invalid_noise_model(self, raw_haemo_with_events) -> None:
        with pytest.raises(ValidationError, match="noise_model"):
            GLMBlock(params=GLMParams(noise_model="bad")).run(
                None, {"upstream": raw_haemo_with_events},
            )

    def test_negative_drift_high_freq(self, raw_haemo_with_events) -> None:
        with pytest.raises(ValidationError, match="drift_high_freq"):
            GLMBlock(params=GLMParams(drift_high_freq=-0.01)).run(
                None, {"upstream": raw_haemo_with_events},
            )

    def test_custom_params(self, raw_haemo_with_events) -> None:
        p = GLMParams(drift_model="polynomial", hrf_model="spm", noise_model="ols")
        result = GLMBlock(params=p).run(None, {"upstream": raw_haemo_with_events})
        assert result.metadata["drift_model"] == "polynomial"

    def test_registry_has_glm(self) -> None:
        from nirspy.blocks import registry
        assert "glm" in registry.list_blocks()


class TestGLMResult:
    def test_construction(self) -> None:
        r = GLMResult(
            theta=np.zeros((3, 4)), t_stats=np.ones((3, 4)),
            p_values=np.full((3, 4), 0.05), mse=np.ones(4),
            channel_names=[f"ch{i}" for i in range(4)],
            regressor_names=[f"reg{i}" for i in range(3)],
            design_matrix=np.zeros((100, 3)), noise_model="ar1",
        )
        assert r.theta.shape == (3, 4)

    def test_to_dataframe_pure(self) -> None:
        r = GLMResult(
            theta=np.arange(6, dtype=float).reshape(2, 3),
            t_stats=np.zeros((2, 3)), p_values=np.zeros((2, 3)),
            mse=np.ones(3), channel_names=["A", "B", "C"],
            regressor_names=["x", "y"], design_matrix=np.zeros((50, 2)),
        )
        df = r.to_dataframe()
        assert len(df) == 6

    def test_get_contrast_pure(self) -> None:
        r = GLMResult(
            theta=np.zeros((2, 3)),
            t_stats=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            p_values=np.zeros((2, 3)), mse=np.ones(3),
            channel_names=["A", "B", "C"],
            regressor_names=["c1", "c2"], design_matrix=np.zeros((50, 2)),
        )
        np.testing.assert_array_equal(r.get_contrast("c2"), [4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# T-040 — GLMParams new fields: condition_durations and per_condition_groups
# ---------------------------------------------------------------------------


class TestGLMParamsT040:
    """Tests for condition_durations and per_condition_groups fields (T-040)."""

    def test_condition_durations_propagated_to_adapter(
        self, raw_haemo_multi_condition: mne.io.BaseRaw
    ) -> None:
        """condition_durations must be forwarded to adapter.run_glm."""
        from unittest.mock import MagicMock

        from nirspy.domain.glm_result import GLMResult

        # Build a minimal GLMResult stub
        stub = GLMResult(
            theta=np.zeros((1, 4)),
            t_stats=np.zeros((1, 4)),
            p_values=np.zeros((1, 4)),
            mse=np.ones(4),
            channel_names=["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"],
            regressor_names=["cond_A"],
            design_matrix=np.zeros((900, 1)),
            metadata={"conditions": ["cond_A"]},
        )

        durations = {"cond_A": 5.0, "cond_B": 3.0}
        params = GLMParams(condition_durations=durations)

        mock_adapter = MagicMock()
        mock_adapter.run_glm.return_value = stub

        GLMBlock(params=params, adapter=mock_adapter).run(
            None, {"upstream": raw_haemo_multi_condition}
        )

        call_kwargs = mock_adapter.run_glm.call_args.kwargs
        assert call_kwargs["condition_durations"] == durations

    def test_per_condition_groups_propagated_to_adapter(
        self, raw_haemo_multi_condition: mne.io.BaseRaw
    ) -> None:
        """per_condition_groups must be forwarded to adapter.run_glm."""
        from unittest.mock import MagicMock

        from nirspy.domain.glm_result import GLMResult

        stub = GLMResult(
            theta=np.zeros((1, 4)),
            t_stats=np.zeros((1, 4)),
            p_values=np.zeros((1, 4)),
            mse=np.ones(4),
            channel_names=["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"],
            regressor_names=["motor"],
            design_matrix=np.zeros((900, 1)),
            metadata={"conditions": ["motor"]},
        )

        groups = {"motor": ["cond_A", "cond_B"]}
        params = GLMParams(per_condition_groups=groups)

        mock_adapter = MagicMock()
        mock_adapter.run_glm.return_value = stub

        GLMBlock(params=params, adapter=mock_adapter).run(
            None, {"upstream": raw_haemo_multi_condition}
        )

        call_kwargs = mock_adapter.run_glm.call_args.kwargs
        assert call_kwargs["per_condition_groups"] == groups

    def test_both_none_retrocompatible(
        self, raw_haemo_multi_condition: mne.io.BaseRaw
    ) -> None:
        """When both new fields are None, adapter.run_glm still receives None for each."""
        from unittest.mock import MagicMock

        from nirspy.domain.glm_result import GLMResult

        stub = GLMResult(
            theta=np.zeros((2, 4)),
            t_stats=np.zeros((2, 4)),
            p_values=np.zeros((2, 4)),
            mse=np.ones(4),
            channel_names=["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"],
            regressor_names=["cond_A", "cond_B"],
            design_matrix=np.zeros((900, 2)),
            metadata={"conditions": ["cond_A", "cond_B"]},
        )

        params = GLMParams()  # both new fields default to None
        mock_adapter = MagicMock()
        mock_adapter.run_glm.return_value = stub

        result = GLMBlock(params=params, adapter=mock_adapter).run(
            None, {"upstream": raw_haemo_multi_condition}
        )

        call_kwargs = mock_adapter.run_glm.call_args.kwargs
        assert call_kwargs["condition_durations"] is None
        assert call_kwargs["per_condition_groups"] is None
        assert result.block_id == "glm"


class TestDataTypeGLMResult:
    def test_enum_value(self) -> None:
        assert DataType.GLM_RESULT == "glm_result"

    def test_block_spec_with_glm_result(self) -> None:
        from nirspy.domain.block import BlockSpec
        s = BlockSpec(
            block_id="t", display_name="T",
            input_type=DataType.RAW_HAEMO, output_type=DataType.GLM_RESULT,
        )
        assert s.output_type == DataType.GLM_RESULT
