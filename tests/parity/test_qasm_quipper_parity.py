"""Temporary parity tests: legacy Quipper path vs new QASM path."""

from __future__ import annotations

from bosonic_model.qasm import Translator

from quipper_distributor.bosonic_adapter import circuit_to_legacy_gates
from quipper_distributor.config import KAHYPAR_CONFIG
from quipper_distributor.dist_circuit_builder import build_circuit
from quipper_distributor.models.gate import get_wires, is_cz
from quipper_distributor.parsing.quipper_ascii import parse_circuit
from quipper_distributor.partitioner import partitioner
from quipper_distributor.preparation import prepare_circuit


def _count_non_local(segments) -> int:
    total = 0
    for seg in segments:
        for g in seg.gates:
            if is_cz(g):
                wires = get_wires(g)
                blocks = {seg.partition.get(w) for w in wires if seg.partition.get(w) is not None}
                total += max(0, len(blocks) - 1)
    return total


def _run_old(quipper_ascii: str, k: int, init_seg_size: int = 1000) -> tuple[int, int, int, int]:
    circuit = parse_circuit(quipper_ascii)
    gates = prepare_circuit(circuit.gates, keep_ccz=False)
    n_wires = len(circuit.inputs)

    segments = partitioner(
        k=k,
        init_seg_size=init_seg_size,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
        n_qubits=n_wires,
        n_wires=n_wires,
        gates=gates,
    )
    _, _, n_ebits, n_teleports = build_circuit(segments, n_wires, n_wires)

    return len(gates), sum(1 for g in gates if is_cz(g)), _count_non_local(segments), n_ebits + n_teleports


def _run_new(qasm_text: str, k: int, init_seg_size: int = 1000) -> tuple[int, int, int, int]:
    circuit = Translator().from_qasm(qasm_text)
    gates = prepare_circuit(circuit_to_legacy_gates(circuit), keep_ccz=False)
    n_wires = circuit.qubits()

    segments = partitioner(
        k=k,
        init_seg_size=init_seg_size,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
        n_qubits=n_wires,
        n_wires=n_wires,
        gates=gates,
    )
    _, _, n_ebits, n_teleports = build_circuit(segments, n_wires, n_wires)

    return len(gates), sum(1 for g in gates if is_cz(g)), _count_non_local(segments), n_ebits + n_teleports


def test_simple_parity(monkeypatch) -> None:
    monkeypatch.setenv("QUIPPER_DISTRIBUTOR_PARTITIONER", "fallback")

    quipper_ascii = """
Inputs: 0:Qbit, 1:Qbit, 2:Qbit, 3:Qbit
QGate[\"not\"](3) with controls=[+1]
QGate[\"not\"](2) with controls=[+1]
QGate[\"not\"](0) with controls=[+1]
Outputs: 0:Qbit, 1:Qbit, 2:Qbit, 3:Qbit
"""

    qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
cx q[1], q[3];
cx q[1], q[2];
cx q[1], q[0];
"""

    assert _run_old(quipper_ascii, k=2) == _run_new(qasm, k=2)


def test_toffoli_parity(monkeypatch) -> None:
    monkeypatch.setenv("QUIPPER_DISTRIBUTOR_PARTITIONER", "fallback")

    quipper_ascii = """
Inputs: 0:Qbit, 1:Qbit, 2:Qbit
QGate[\"not\"](2) with controls=[+0, +1]
Outputs: 0:Qbit, 1:Qbit, 2:Qbit
"""

    qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
ccx q[0], q[1], q[2];
"""

    assert _run_old(quipper_ascii, k=2) == _run_new(qasm, k=2)
