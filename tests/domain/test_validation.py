"""Tests for nirspy.domain.validation.validate_io_chain."""

from __future__ import annotations

import pytest

from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError
from nirspy.domain.validation import validate_io_chain
from tests.conftest import make_block


class TestValidateIoChainValid:
    """Cases where validate_io_chain should NOT raise."""

    def test_empty_list_is_valid(self) -> None:
        validate_io_chain([])  # must not raise

    def test_single_block_is_valid(self) -> None:
        block = make_block("only", DataType.RAW, DataType.RAW_OD)
        validate_io_chain([block])  # must not raise

    def test_compatible_chain_raw_to_od_to_haemo(
        self,
        fake_block_raw_to_od,  # noqa: ANN001
        fake_block_od_to_haemo,  # noqa: ANN001
    ) -> None:
        validate_io_chain([fake_block_raw_to_od, fake_block_od_to_haemo])

    def test_three_blocks_compatible(self) -> None:
        b1 = make_block("b1", DataType.ANY, DataType.RAW)
        b2 = make_block("b2", DataType.RAW, DataType.RAW_OD)
        b3 = make_block("b3", DataType.RAW_OD, DataType.RAW_HAEMO)
        validate_io_chain([b1, b2, b3])

    def test_any_wildcard_on_producer_skips_check(self) -> None:
        producer = make_block("source", DataType.RAW, DataType.ANY)
        consumer = make_block("dest", DataType.RAW_HAEMO, DataType.EVOKED)
        validate_io_chain([producer, consumer])

    def test_any_wildcard_on_consumer_skips_check(self) -> None:
        producer = make_block("source", DataType.RAW, DataType.RAW_OD)
        consumer = make_block("dest", DataType.ANY, DataType.RAW_HAEMO)
        validate_io_chain([producer, consumer])

    def test_disabled_blocks_are_excluded_from_check(self) -> None:
        """Disabled blocks are skipped, so an incompatible disabled block must not fail."""
        b1 = make_block("b1", DataType.RAW, DataType.RAW_OD)
        bad = make_block("bad", DataType.EVOKED, DataType.EPOCHS, enabled=False)
        b3 = make_block("b3", DataType.RAW_OD, DataType.RAW_HAEMO)
        validate_io_chain([b1, bad, b3])

    def test_all_same_type_chain(self) -> None:
        blocks = [make_block(f"b{i}", DataType.RAW, DataType.RAW) for i in range(5)]
        validate_io_chain(blocks)


class TestValidateIoChainInvalid:
    """Cases where validate_io_chain should raise ValidationError."""

    def test_incompatible_pair_raises(self) -> None:
        b1 = make_block("b1", DataType.RAW, DataType.RAW_OD)
        b2 = make_block("b2", DataType.RAW_HAEMO, DataType.EVOKED)
        with pytest.raises(ValidationError, match="b1"):
            validate_io_chain([b1, b2])

    def test_error_message_mentions_both_block_ids(self) -> None:
        producer = make_block("raw_loader", DataType.RAW, DataType.RAW_OD)
        consumer = make_block("beer_lambert", DataType.EPOCHS, DataType.EVOKED)
        with pytest.raises(ValidationError) as exc_info:
            validate_io_chain([producer, consumer])
        msg = str(exc_info.value)
        assert "raw_loader" in msg
        assert "beer_lambert" in msg

    def test_incompatible_in_middle_of_chain(self) -> None:
        b1 = make_block("b1", DataType.RAW, DataType.RAW_OD)
        b2 = make_block("b2", DataType.RAW_OD, DataType.RAW_HAEMO)
        b3 = make_block("b3", DataType.EPOCHS, DataType.EVOKED)  # incompatible input
        with pytest.raises(ValidationError):
            validate_io_chain([b1, b2, b3])

    def test_only_enabled_blocks_considered(self) -> None:
        """With all blocks disabled, chain should be valid (nothing to check)."""
        b1 = make_block("b1", DataType.RAW, DataType.RAW_OD, enabled=False)
        b2 = make_block("b2", DataType.EVOKED, DataType.EPOCHS, enabled=False)
        validate_io_chain([b1, b2])  # must not raise — both disabled


class TestSourceBlockValidation:
    """Cases for DataType.NONE (source block) positional constraint."""

    def test_source_block_at_position_0_is_valid(self) -> None:
        source = make_block("load_snirf", DataType.NONE, DataType.RAW)
        consumer = make_block("od", DataType.RAW, DataType.RAW_OD)
        validate_io_chain([source, consumer])  # must not raise

    def test_source_block_alone_is_valid(self) -> None:
        source = make_block("load_snirf", DataType.NONE, DataType.RAW)
        validate_io_chain([source])  # must not raise

    def test_source_block_at_position_1_raises(self) -> None:
        b1 = make_block("od", DataType.RAW, DataType.RAW_OD)
        source = make_block("load_snirf", DataType.NONE, DataType.RAW)
        with pytest.raises(ValidationError, match="load_snirf"):
            validate_io_chain([b1, source])

    def test_source_block_at_position_2_raises(self) -> None:
        b1 = make_block("b1", DataType.RAW, DataType.RAW_OD)
        b2 = make_block("b2", DataType.RAW_OD, DataType.RAW_HAEMO)
        source = make_block("load_snirf", DataType.NONE, DataType.RAW)
        with pytest.raises(ValidationError, match="position 2"):
            validate_io_chain([b1, b2, source])

    def test_source_block_error_message_mentions_first(self) -> None:
        b1 = make_block("od", DataType.RAW, DataType.RAW_OD)
        source = make_block("load_snirf", DataType.NONE, DataType.RAW)
        with pytest.raises(ValidationError, match="first step"):
            validate_io_chain([b1, source])

    def test_disabled_source_at_position_1_skipped(self) -> None:
        """Disabled source blocks are excluded from the enabled list."""
        b1 = make_block("od", DataType.RAW, DataType.RAW_OD)
        source = make_block("load_snirf", DataType.NONE, DataType.RAW, enabled=False)
        b3 = make_block("b3", DataType.RAW_OD, DataType.RAW_HAEMO)
        validate_io_chain([b1, source, b3])  # disabled source skipped

    def test_any_blocks_unaffected_by_none_logic(self) -> None:
        """BandpassFilter (ANY/ANY) must still work normally."""
        source = make_block("load", DataType.NONE, DataType.RAW)
        bpf = make_block("bandpass", DataType.ANY, DataType.ANY)
        consumer = make_block("mbll", DataType.RAW_OD, DataType.RAW_HAEMO)
        validate_io_chain([source, bpf, consumer])  # must not raise

    def test_none_output_skips_type_check(self) -> None:
        """Future sink block with output_type=NONE should not cause type error."""
        source = make_block("load", DataType.NONE, DataType.RAW)
        sink = make_block("export", DataType.RAW, DataType.NONE)
        validate_io_chain([source, sink])  # must not raise
