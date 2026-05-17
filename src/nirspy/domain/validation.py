"""Pipeline validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nirspy.domain.block import Block

from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError


def validate_io_chain(steps: list[Block]) -> None:
    """Verify that output types chain correctly through *steps*.

    Only *enabled* blocks are checked. ``DataType.ANY`` on either side
    of a connection skips the type check for that edge.

    Raises
    ------
    ValidationError
        When an output type does not match the next block's expected input type.
    """
    enabled = [s for s in steps if s.spec.enabled]

    for i in range(len(enabled) - 1):
        producer = enabled[i]
        consumer = enabled[i + 1]

        out_type = producer.spec.output_type
        in_type = consumer.spec.input_type

        if out_type is DataType.ANY or in_type is DataType.ANY:
            continue

        if out_type != in_type:
            raise ValidationError(
                f"Type mismatch between '{producer.spec.block_id}' (output: {out_type.value}) "
                f"and '{consumer.spec.block_id}' (input: {in_type.value})"
            )
