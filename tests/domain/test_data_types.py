"""Tests for nirspy.domain.data_types."""

from __future__ import annotations

import pytest

from nirspy.domain.data_types import DataType


class TestDataTypeEnum:
    """DataType enum invariants."""

    def test_all_expected_members_exist(self) -> None:
        names = {m.name for m in DataType}
        expected = {"NONE", "RAW", "RAW_OD", "RAW_HAEMO", "EPOCHS", "EVOKED", "DATAFRAME", "ANY", "GLM_RESULT"}
        assert names == expected

    def test_values_are_lowercase_strings(self) -> None:
        for member in DataType:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    def test_str_mixin_equality(self) -> None:
        """str mixin: DataType value equals its string representation."""
        assert DataType.RAW == "raw"
        assert DataType.RAW_OD == "raw_od"
        assert DataType.RAW_HAEMO == "raw_haemo"

    def test_members_are_hashable(self) -> None:
        seen = {DataType.RAW, DataType.RAW_OD}
        assert DataType.RAW in seen

    def test_any_wildcard_exists(self) -> None:
        assert DataType.ANY.value == "any"

    def test_json_serialisable_via_value(self) -> None:
        import json

        payload = {"type": DataType.EVOKED.value}
        assert json.dumps(payload) == '{"type": "evoked"}'

    def test_str_coercion(self) -> None:
        """str(DataType.RAW) must include 'raw'."""
        assert "raw" in str(DataType.RAW).lower()


    def test_none_member_exists(self) -> None:
        assert DataType.NONE.value == "none"

    def test_none_is_not_any(self) -> None:
        """NONE and ANY are semantically distinct."""
        assert DataType.NONE is not DataType.ANY
        assert DataType.NONE != DataType.ANY
    @pytest.mark.parametrize(
        "member,expected_value",
        [
            (DataType.NONE, "none"),
            (DataType.RAW, "raw"),
            (DataType.RAW_OD, "raw_od"),
            (DataType.RAW_HAEMO, "raw_haemo"),
            (DataType.EPOCHS, "epochs"),
            (DataType.EVOKED, "evoked"),
            (DataType.DATAFRAME, "dataframe"),
            (DataType.ANY, "any"),
            (DataType.GLM_RESULT, "glm_result"),
        ],
    )
    def test_member_values(self, member: DataType, expected_value: str) -> None:
        assert member.value == expected_value
