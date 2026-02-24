"""Integration tests for QASM + bosonic_model pipeline."""

from __future__ import annotations

import pytest


def _run_pipeline(qasm_text: str, k: int, init_seg_size: int = 1000) -> tuple[int, int, int]:
    from bosonic_model.qasm import Translator

    from quipper_distributor.bosonic_adapter import circuit_to_legacy_gates
    from quipper_distributor.config import KAHYPAR_CONFIG
    from quipper_distributor.dist_circuit_builder import build_circuit
    from quipper_distributor.models.gate import is_cz
    from quipper_distributor.partitioner import partitioner
    from quipper_distributor.preparation import prepare_circuit

    circuit = Translator().from_qasm(qasm_text)
    gates = prepare_circuit(circuit_to_legacy_gates(circuit), keep_ccz=False)
    n_qubits = circuit.qubits()

    segments = partitioner(
        k=k,
        init_seg_size=init_seg_size,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
        n_qubits=n_qubits,
        n_wires=n_qubits,
        gates=gates,
    )

    _, _, n_ebits, n_teleports = build_circuit(segments, n_qubits, n_qubits)
    cz_count = sum(1 for g in gates if is_cz(g))
    return cz_count, n_ebits, n_teleports


@pytest.mark.integration
def test_simple_qasm_k2() -> None:
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[4];

    cx q[1], q[3];
    cx q[1], q[2];
    cx q[1], q[0];
    """

    cz_count, n_ebits, n_teleports = _run_pipeline(qasm, k=2)

    assert cz_count > 0
    assert n_ebits >= 0
    assert n_teleports >= 0


@pytest.mark.integration
def test_toffoli_qasm_k2() -> None:
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[3];

    ccx q[0], q[1], q[2];
    """

    cz_count, n_ebits, n_teleports = _run_pipeline(qasm, k=2)

    assert cz_count > 0
    assert n_ebits >= 0
    assert n_teleports >= 0


@pytest.mark.integration
def test_output_stats_non_negative() -> None:
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[2];

    h q[0];
    cx q[0], q[1];
    """

    _, n_ebits, n_teleports = _run_pipeline(qasm, k=2)

    assert isinstance(n_ebits, int) and n_ebits >= 0
    assert isinstance(n_teleports, int) and n_teleports >= 0
