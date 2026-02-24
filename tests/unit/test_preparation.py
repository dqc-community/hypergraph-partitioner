"""Unit tests for the gate normalisation pipeline."""

from __future__ import annotations

import pytest

from quipper_distributor.models.gate import QGate, SignedWire, is_cz
from quipper_distributor.preparation import (
    decompose_toffoli,
    prepare_circuit,
    push_single_qubit_gates,
    remove_swaps,
    to_controlled_z,
)


class TestRemoveSwaps:
    def test_swap_becomes_three_cnots(self):
        g = QGate(name="swap", inputs=[0, 1])
        result = remove_swaps([g])
        assert len(result) == 3
        for gate in result:
            assert isinstance(gate, QGate)
            assert gate.name == "not"
            assert len(gate.controls) == 1

    def test_non_swap_unchanged(self):
        g = QGate(name="H", inputs=[0])
        result = remove_swaps([g])
        assert result == [g]

    def test_swap_wires_correct(self):
        g = QGate(name="swap", inputs=[2, 3])
        result = remove_swaps([g])
        # CNOT(2, ctrl=3), CNOT(3, ctrl=2), CNOT(2, ctrl=3)
        assert result[0].inputs == [2] and result[0].controls[0].wire == 3
        assert result[1].inputs == [3] and result[1].controls[0].wire == 2
        assert result[2].inputs == [2] and result[2].controls[0].wire == 3


class TestToControlledZ:
    def test_cnot_to_h_cz_h(self):
        g = QGate(name="not", inputs=[0], controls=[SignedWire(wire=1, positive=True)])
        result = to_controlled_z([g])
        assert len(result) == 3
        assert isinstance(result[0], QGate) and result[0].name == "H"
        assert isinstance(result[1], QGate) and result[1].name == "CZ"
        assert isinstance(result[2], QGate) and result[2].name == "H"
        assert result[0].inputs == [0]
        assert result[1].inputs == [0]
        assert result[1].controls[0].wire == 1
        assert result[2].inputs == [0]

    def test_h_gate_unchanged(self):
        g = QGate(name="H", inputs=[0])
        result = to_controlled_z([g])
        assert result == [g]

    def test_multi_control_not_converted(self):
        # Two-control 'not' is not converted by to_controlled_z (only single control)
        g = QGate(
            name="not",
            inputs=[0],
            controls=[
                SignedWire(wire=1, positive=True),
                SignedWire(wire=2, positive=True),
            ],
        )
        result = to_controlled_z([g])
        assert result == [g]


class TestDecomposeToffoli:
    def test_toffoli_decomposes(self):
        g = QGate(
            name="not",
            inputs=[0],
            controls=[
                SignedWire(wire=1, positive=True),
                SignedWire(wire=2, positive=True),
            ],
        )
        result = decompose_toffoli([g])
        # Should produce more than 1 gate
        assert len(result) > 3
        # Should contain only single- and two-qubit gates
        for gate in result:
            assert isinstance(gate, QGate)
            assert len(gate.controls) <= 1

    def test_single_control_not_decomposes(self):
        # Single-control CNOT should not be decomposed by this step
        g = QGate(name="not", inputs=[0], controls=[SignedWire(wire=1, positive=True)])
        result = decompose_toffoli([g])
        assert result == [g]

    def test_negative_control_toffoli(self):
        g = QGate(
            name="not",
            inputs=[0],
            controls=[
                SignedWire(wire=1, positive=True),
                SignedWire(wire=2, positive=False),
            ],
        )
        result = decompose_toffoli([g])
        # Should add X gates around the decomposition for wire 2
        assert len(result) > 1
        x_gates = [r for r in result if isinstance(r, QGate) and r.name == "X" and r.inputs == [2]]
        assert len(x_gates) >= 2  # X before and after


class TestPushSingleQubitGates:
    def test_hh_cancels(self):
        """Two adjacent H gates on the same wire cancel."""
        gates = [
            QGate(name="H", inputs=[0]),
            QGate(name="H", inputs=[0]),
        ]
        result = push_single_qubit_gates(gates)
        h_gates = [g for g in result if isinstance(g, QGate) and g.name == "H" and g.inputs == [0]]
        assert len(h_gates) == 0

    def test_x_pushed_through_cz_generates_z_byproduct(self):
        """An X gate on one wire, followed by a CZ, should generate a Z on the other."""
        gates = [
            QGate(name="X", inputs=[0]),
            QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)]),
        ]
        result = push_single_qubit_gates(gates)
        gate_names = [g.name for g in result if isinstance(g, QGate)]
        assert "Z" in gate_names

    def test_non_interfering_gates_preserved(self):
        """S and T gates should pass through unchanged."""
        gates = [
            QGate(name="S", inputs=[0]),
            QGate(name="T", inputs=[1]),
        ]
        result = push_single_qubit_gates(gates)
        names = [g.name for g in result if isinstance(g, QGate)]
        assert "S" in names
        assert "T" in names


class TestPrepareCircuit:
    def test_simple_pipeline(self):
        """Full pipeline on simple circuit: 3 CNOTs become CZ-based."""
        from quipper_distributor.parsing.quipper_ascii import parse_circuit

        text = (
            "Inputs: 0:Qbit, 1:Qbit, 2:Qbit, 3:Qbit\n"
            'QGate["not"](3) with controls=[+1]\n'
            'QGate["not"](2) with controls=[+1]\n'
            'QGate["not"](0) with controls=[+1]\n'
            "Outputs: 0:Qbit, 1:Qbit, 2:Qbit, 3:Qbit\n"
        )
        c = parse_circuit(text)
        result = prepare_circuit(c.gates, keep_ccz=False)
        # Should have CZ gates
        cz_gates = [g for g in result if is_cz(g)]
        assert len(cz_gates) == 3

    def test_toffoli_pipeline(self):
        """CCX in the circuit gets decomposed when keep_ccz=False."""
        from quipper_distributor.parsing.quipper_ascii import parse_circuit

        text = (
            "Inputs: 0:Qbit, 1:Qbit, 2:Qbit\n"
            'QGate["not"](0) with controls=[+1, +2]\n'
            "Outputs: 0:Qbit, 1:Qbit, 2:Qbit\n"
        )
        c = parse_circuit(text)
        result = prepare_circuit(c.gates, keep_ccz=False)
        # Toffoli should be decomposed — no 2-control gates remain
        two_ctrl = [g for g in result if isinstance(g, QGate) and len(g.controls) == 2]
        assert len(two_ctrl) == 0
