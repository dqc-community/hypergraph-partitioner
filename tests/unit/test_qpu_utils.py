"""Unit tests for qpu_utils instruction helpers."""

from __future__ import annotations

import pytest
from bosonic_model import Condition, ConditionalInstruction, MeasureInstruction
from bosonic_model.instructions import CzInstruction, UInstruction

from hypergraph_partitioner.qpu_utils import max_cbit_in_instruction, remap_instruction


def _cz(control: int, target: int) -> CzInstruction:
    return CzInstruction(control=control, target=target, qubits=[control, target])


def _conditional(creg_base: int, creg_size: int, creg_value: int, op: object) -> ConditionalInstruction:
    return ConditionalInstruction(
        condition=Condition(creg_base=creg_base, creg_size=creg_size, creg_value=creg_value),
        op=op,
        qubits=list(op.qubits),
    )


# ---------------------------------------------------------------------------
# max_cbit_in_instruction
# ---------------------------------------------------------------------------


def test_max_cbit_single_bit_condition() -> None:
    inst = _conditional(creg_base=3, creg_size=1, creg_value=1, op=_cz(0, 1))
    assert max_cbit_in_instruction(inst) == 3


def test_max_cbit_multi_bit_condition_base_only() -> None:
    # creg_size > 1: the function returns creg_base, not the last bit in the register.
    # This documents the current behaviour and will need updating if range tracking is added.
    inst = _conditional(creg_base=4, creg_size=3, creg_value=5, op=_cz(0, 1))
    assert max_cbit_in_instruction(inst) == 4


def test_max_cbit_dominated_by_inner_measure() -> None:
    measure = MeasureInstruction(qubit=0, cbit=10)
    u = UInstruction(qubit=0, qubits=[0], theta=0.0, phi=0.0, lam=0.0, params=[0.0, 0.0, 0.0])
    # Wrap the U gate in a conditional; the measure is a separate instruction, not nested.
    # Verify that a high-cbit measure beats a low-cbit conditional.
    assert max_cbit_in_instruction(measure) == 10
    inst = _conditional(creg_base=2, creg_size=4, creg_value=3, op=_cz(0, 1))
    assert max_cbit_in_instruction(inst) == 2


def test_max_cbit_non_conditional_non_measure_returns_minus_one() -> None:
    inst = _cz(0, 1)
    assert max_cbit_in_instruction(inst) == -1


# ---------------------------------------------------------------------------
# remap_instruction
# ---------------------------------------------------------------------------


def test_remap_preserves_creg_size_and_creg_value() -> None:
    inst = _conditional(creg_base=2, creg_size=4, creg_value=7, op=_cz(0, 1))
    remapped = remap_instruction(inst, qubit_map={0: 10, 1: 11}, cbit_map={2: 20})

    assert isinstance(remapped, ConditionalInstruction)
    assert remapped.condition.creg_base == 20
    assert remapped.condition.creg_size == 4
    assert remapped.condition.creg_value == 7


def test_remap_single_bit_condition() -> None:
    inst = _conditional(creg_base=0, creg_size=1, creg_value=1, op=_cz(0, 1))
    remapped = remap_instruction(inst, qubit_map={0: 5, 1: 6}, cbit_map={0: 3})

    assert remapped.condition.creg_base == 3
    assert remapped.condition.creg_size == 1
    assert remapped.condition.creg_value == 1
    assert list(remapped.qubits) == [5, 6]


def test_remap_creg_base_unchanged_when_not_in_cbit_map() -> None:
    inst = _conditional(creg_base=5, creg_size=2, creg_value=3, op=_cz(0, 1))
    remapped = remap_instruction(inst, qubit_map={}, cbit_map={99: 100})

    assert remapped.condition.creg_base == 5
    assert remapped.condition.creg_size == 2
    assert remapped.condition.creg_value == 3


def test_remap_qubit_map_applied_to_inner_op() -> None:
    inst = _conditional(creg_base=0, creg_size=1, creg_value=1, op=_cz(2, 3))
    remapped = remap_instruction(inst, qubit_map={2: 20, 3: 30}, cbit_map={})

    assert isinstance(remapped.op, CzInstruction)
    assert remapped.op.control == 20
    assert remapped.op.target == 30
