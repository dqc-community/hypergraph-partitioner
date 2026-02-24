"""Unit tests for bosonic_model -> legacy gate adapter."""

from __future__ import annotations

from bosonic_model.qasm import Translator

from quipper_distributor.bosonic_adapter import circuit_to_legacy_gates
from quipper_distributor.models.gate import QGate, QMeas


def test_cx_maps_to_not_with_control() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        cx q[0], q[1];
        """
    )

    gates = circuit_to_legacy_gates(circuit)

    assert len(gates) == 1
    gate = gates[0]
    assert isinstance(gate, QGate)
    assert gate.name == "not"
    assert gate.inputs == [1]
    assert gate.controls[0].wire == 0


def test_ccx_maps_to_two_control_not() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        ccx q[0], q[1], q[2];
        """
    )

    gates = circuit_to_legacy_gates(circuit)

    assert len(gates) == 1
    gate = gates[0]
    assert isinstance(gate, QGate)
    assert gate.name == "not"
    assert len(gate.controls) == 2


def test_measure_maps_to_qmeas() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[1];
        creg c[1];
        measure q[0] -> c[0];
        """
    )

    gates = circuit_to_legacy_gates(circuit)

    assert len(gates) == 1
    assert isinstance(gates[0], QMeas)
