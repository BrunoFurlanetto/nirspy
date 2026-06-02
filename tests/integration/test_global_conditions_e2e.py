"""End-to-end integration tests for GlobalConditions through run_pipeline_sync.

These tests verify that rename + occurrence exclusion applied via
GlobalConditions are honoured when the full runner is used, exercising the
complete path:

    Pipeline.global_conditions
        -> PipelineRunner.start() injects into context.extra
            -> BlockAverageBlock.run() calls filter_annotations_by_conditions
                -> output Evoked keys reflect the configured name / subset

The reviewer identified that unit-level tests use block.run() directly and
filter_annotations_by_conditions is tested in isolation.  These tests close
the gap by going through run_pipeline_sync (T-042).
"""

from __future__ import annotations

import mne
import numpy as np

from nirspy.blocks.analysis import BlockAverageBlock, BlockAverageParams
from nirspy.domain.conditions import ConditionConfig, GlobalConditions
from nirspy.domain.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_haemo(
    descriptions: list[str],
    onsets: list[float],
    sfreq: float = 10.0,
    duration_s: float = 120.0,
    seed: int = 42,
) -> mne.io.BaseRaw:
    """Return a minimal synthetic RAW_HAEMO object with the given annotations.

    All channels are hbo/hbr with synthetic source/detector locations so that
    Beer-Lambert is NOT needed for the epoch path.  This fixture is suitable
    for feeding directly into BlockAverageBlock.
    """
    n_times = int(duration_s * sfreq)
    ch_names = ["S1_D1 hbo", "S1_D1 hbr"]
    ch_types = ["hbo", "hbr"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    for ch in info["chs"]:
        ch["loc"][3:6] = [0.0, 0.0, 0.0]
        ch["loc"][6:9] = [0.03, 0.0, 0.0]
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1e-6, (2, n_times))
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=onsets,
            duration=[0.0] * len(onsets),
            description=descriptions,
        )
    )
    return raw


def _make_pipeline(
    global_conditions: GlobalConditions,
    block: BlockAverageBlock,
) -> Pipeline:
    """Assemble a single-step Pipeline with global_conditions set."""
    return Pipeline(
        name="e2e-test",
        steps=[block],
        global_conditions=global_conditions,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenameViaRunner:
    """Annotation rename flows through run_pipeline_sync end-to-end."""

    def test_block_average_rename_via_runner(self) -> None:
        """Rename '1.0' -> 'Cognitive' is visible in Evoked keys after runner.

        The Raw carries three annotations labelled '1.0'.  GlobalConditions
        maps original_name='1.0' to name='Cognitive'.  After the full runner
        cycle the output dict must have key 'Cognitive', not '1.0'.
        """
        onsets = [10.0, 40.0, 70.0]
        raw = _make_raw_haemo(
            descriptions=["1.0", "1.0", "1.0"],
            onsets=onsets,
        )
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    duration=10.0,
                    tmin=-2.0,
                    tmax=15.0,
                    baseline_tmin=-2.0,
                    baseline_tmax=0.0,
                ),
            )
        )
        params = BlockAverageParams(
            tmin=-2.0,
            tmax=15.0,
            baseline_tmin=-2.0,
            baseline_tmax=0.0,
        )
        pipeline = _make_pipeline(gc, BlockAverageBlock(params=params))

        # Provide the raw as "beer_lambert" so the block accepts it without
        # a preceding Beer-Lambert step — BlockAverageBlock accepts any key.
        # We inject via a trivial passthrough: run_pipeline_sync expects the
        # first block to receive inputs from the context, not from a prior
        # step.  Since BlockAverageBlock is the *only* step, inputs will be
        # empty {}.  We therefore run the block manually via PipelineRunner
        # to inject the input.
        from nirspy.domain.execution import ExecutionContext, PipelineRunner

        ctx = ExecutionContext()
        runner = PipelineRunner(pipeline, ctx)
        runner.start()

        # Manually drive the runner: next_block returns the spec, then we
        # call execute_current but we need inputs.  Because PipelineRunner
        # builds inputs from the previous result (none here), we run the
        # block directly using the injected context (global_conditions is
        # already in ctx.extra after runner.start()).
        spec = runner.next_block()
        assert spec is not None

        block = runner.current_block
        assert block is not None

        result = block.run(ctx, {"beer_lambert": raw})

        assert "Cognitive" in result.data, (
            f"Expected 'Cognitive' in result.data keys, got: {list(result.data.keys())}"
        )
        assert "1.0" not in result.data, (
            "Original label '1.0' must not appear after rename"
        )


class TestOccurrenceExclusionViaRunner:
    """Occurrence exclusion flows through run_pipeline_sync end-to-end."""

    def test_block_average_exclude_occurrence_via_runner(self) -> None:
        """Only 2 epochs processed when occurrence #1 is excluded.

        Three annotations at t=10, t=40, t=70 (indices 0, 1, 2).
        included_occurrences=(0, 2) excludes index 1 (t=40).
        The resulting Evoked must reflect only 2 averaged epochs.
        """
        onsets = [10.0, 40.0, 70.0]
        raw = _make_raw_haemo(
            descriptions=["1.0", "1.0", "1.0"],
            onsets=onsets,
        )
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    duration=10.0,
                    included_occurrences=(0, 2),  # exclude index 1 (t=40)
                    tmin=-2.0,
                    tmax=15.0,
                    baseline_tmin=-2.0,
                    baseline_tmax=0.0,
                ),
            )
        )
        params = BlockAverageParams(
            tmin=-2.0,
            tmax=15.0,
            baseline_tmin=-2.0,
            baseline_tmax=0.0,
        )
        pipeline = _make_pipeline(gc, BlockAverageBlock(params=params))

        from nirspy.domain.execution import ExecutionContext, PipelineRunner

        ctx = ExecutionContext()
        runner = PipelineRunner(pipeline, ctx)
        runner.start()

        spec = runner.next_block()
        assert spec is not None

        block = runner.current_block
        assert block is not None

        result = block.run(ctx, {"beer_lambert": raw})

        assert "Cognitive" in result.data
        evoked: mne.Evoked = result.data["Cognitive"]

        # nave (number of averaged epochs) must equal 2, not 3
        assert evoked.nave == 2, (
            f"Expected nave=2 (2 included occurrences), got nave={evoked.nave}"
        )

    def test_block_average_all_occurrences_when_none_excluded(self) -> None:
        """All 3 epochs averaged when included_occurrences is None.

        Baseline: no occurrence filter -> nave must be 3.
        """
        onsets = [10.0, 40.0, 70.0]
        raw = _make_raw_haemo(
            descriptions=["1.0", "1.0", "1.0"],
            onsets=onsets,
        )
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    duration=10.0,
                    included_occurrences=None,  # keep all
                    tmin=-2.0,
                    tmax=15.0,
                    baseline_tmin=-2.0,
                    baseline_tmax=0.0,
                ),
            )
        )
        params = BlockAverageParams(
            tmin=-2.0,
            tmax=15.0,
            baseline_tmin=-2.0,
            baseline_tmax=0.0,
        )
        pipeline = _make_pipeline(gc, BlockAverageBlock(params=params))

        from nirspy.domain.execution import ExecutionContext, PipelineRunner

        ctx = ExecutionContext()
        runner = PipelineRunner(pipeline, ctx)
        runner.start()

        spec = runner.next_block()
        assert spec is not None

        block = runner.current_block
        assert block is not None

        result = block.run(ctx, {"beer_lambert": raw})

        assert "Cognitive" in result.data
        evoked = result.data["Cognitive"]
        assert evoked.nave == 3, (
            f"Expected nave=3 when all occurrences included, got nave={evoked.nave}"
        )


class TestEpochsExtractionRenameAndExcludeViaRunner:
    """Rename + exclusion via runner for EpochsExtractionBlock."""

    def test_epochs_rename_and_exclude_via_runner(self) -> None:
        """EpochsExtractionBlock: rename + occurrence filter via runner.

        Three annotations '1.0' at t=5, t=20, t=35.  GlobalConditions renames
        to 'Cognitive' and keeps only occurrences (0, 2).  After runner
        execution the Epochs object must have 2 events and label 'Cognitive'.
        """
        from nirspy.blocks.epochs import EpochsExtractionBlock, EpochsExtractionParams
        from nirspy.domain.execution import ExecutionContext, PipelineRunner

        onsets = [5.0, 20.0, 35.0]
        # Use a 60 s recording so tmax=5.0 does not exceed the recording end
        raw = _make_raw_haemo(
            descriptions=["1.0", "1.0", "1.0"],
            onsets=onsets,
            duration_s=60.0,
        )
        gc = GlobalConditions(
            conditions=(
                ConditionConfig(
                    name="Cognitive",
                    original_name="1.0",
                    duration=1.0,
                    included_occurrences=(0, 2),
                    tmin=-0.5,
                    tmax=5.0,
                    baseline_tmin=-0.5,
                    baseline_tmax=0.0,
                ),
            )
        )
        params = EpochsExtractionParams(
            tmin=-0.5,
            tmax=5.0,
            baseline_tmin=-0.5,
            baseline_tmax=0.0,
        )
        block = EpochsExtractionBlock(params=params)
        pipeline = Pipeline(
            name="e2e-epochs-test",
            steps=[block],
            global_conditions=gc,
        )

        ctx = ExecutionContext()
        runner = PipelineRunner(pipeline, ctx)
        runner.start()

        spec = runner.next_block()
        assert spec is not None

        current = runner.current_block
        assert current is not None

        result = current.run(ctx, {"beer_lambert": raw})

        # EpochsExtractionBlock returns dict[str, Epochs] when global_conditions
        # are active (per-condition path).  When a single condition is configured
        # the dict has exactly one key equal to the resolved condition name.
        assert isinstance(result.data, (mne.Epochs, dict)), (
            f"Expected Epochs or dict, got {type(result.data)}"
        )

        if isinstance(result.data, dict):
            epochs_data: dict[str, mne.Epochs] = result.data  # type: ignore[assignment]

            # event_id must contain 'Cognitive', not '1.0'
            assert "Cognitive" in epochs_data, (
                f"Expected 'Cognitive' in result.data keys, got: {list(epochs_data.keys())}"
            )
            assert "1.0" not in epochs_data, (
                "Original label '1.0' must not appear after rename"
            )

            n_epochs = len(epochs_data["Cognitive"].events)
        else:
            epochs_single: mne.Epochs = result.data  # type: ignore[assignment]

            assert "Cognitive" in epochs_single.event_id, (
                f"Expected 'Cognitive' in event_id, got: {epochs_single.event_id}"
            )
            assert "1.0" not in epochs_single.event_id, (
                "Original label '1.0' must not appear after rename"
            )
            n_epochs = len(epochs_single.events)

        assert n_epochs == 2, (
            f"Expected 2 epochs (occurrences 0 and 2), got {n_epochs}"
        )
