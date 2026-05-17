"""Tests for yaml_serializer — dump_pipeline overwrite behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from nirspy.domain.data_types import DataType
from nirspy.domain.pipeline import Pipeline
from nirspy.io.yaml_serializer import dump_pipeline
from tests.conftest import make_block

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_pipeline() -> Pipeline:
    """Create a minimal pipeline for serialisation tests."""
    block = make_block("test_block", DataType.RAW, DataType.RAW)
    return Pipeline(name="test-pipe", description="test", steps=[block])


# ---------------------------------------------------------------------------
# Tests — dump_pipeline overwrite guard (S-03)
# ---------------------------------------------------------------------------


class TestDumpPipelineOverwrite:
    """Tests for the overwrite parameter added in S-03."""

    def test_dump_pipeline_writes_when_file_absent(self, tmp_path: Path) -> None:
        """Default behaviour: writing to a new path succeeds."""
        pipeline = _make_simple_pipeline()
        target = tmp_path / "new_pipeline.yml"

        dump_pipeline(pipeline, target)

        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "test-pipe" in content

    def test_dump_pipeline_raises_when_file_exists_and_overwrite_false(
        self, tmp_path: Path
    ) -> None:
        """FileExistsError raised when path exists and overwrite=False (default)."""
        pipeline = _make_simple_pipeline()
        target = tmp_path / "existing.yml"
        target.write_text("pre-existing content", encoding="utf-8")

        with pytest.raises(FileExistsError, match="already exists"):
            dump_pipeline(pipeline, target)

        # File must remain unchanged
        assert target.read_text(encoding="utf-8") == "pre-existing content"

    def test_dump_pipeline_overwrites_when_overwrite_true(
        self, tmp_path: Path
    ) -> None:
        """File is replaced when overwrite=True."""
        pipeline = _make_simple_pipeline()
        target = tmp_path / "overwrite_me.yml"
        target.write_text("old content", encoding="utf-8")

        dump_pipeline(pipeline, target, overwrite=True)

        content = target.read_text(encoding="utf-8")
        assert "old content" not in content
        assert "test-pipe" in content
