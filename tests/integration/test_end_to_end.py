"""Integration tests (require kahypar installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

CIRCUITS_DIR = Path(__file__).parent.parent.parent / "circuits"
KAHYPAR_CONFIG = str(Path(__file__).parent.parent.parent / "kahypar" / "config" / "km1_kKaHyPar_sea20.ini")


@pytest.mark.integration
def test_simple_circuit_k2():
    """End-to-end: simple circuit with k=2."""
    from quipper_distributor.dist_circuit_builder import build_circuit
    from quipper_distributor.parsing.quipper_ascii import parse_circuit
    from quipper_distributor.partitioner import partitioner
    from quipper_distributor.preparation import prepare_circuit

    text = (CIRCUITS_DIR / "simple").read_text()
    circuit = parse_circuit(text)
    gates = prepare_circuit(circuit.gates, keep_ccz=False)
    n_qubits = len(circuit.inputs)

    segments = partitioner(
        k=2,
        init_seg_size=1000,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
        n_qubits=n_qubits,
        n_wires=n_qubits,
        gates=gates,
    )

    result_gates, new_wires, n_ebits, n_teleports = build_circuit(segments, n_qubits, n_qubits)

    assert len(result_gates) > 0
    assert n_ebits >= 0
    assert n_teleports >= 0


@pytest.mark.integration
def test_qft20_k3():
    """End-to-end: QFT-20 circuit with k=3."""
    from quipper_distributor.dist_circuit_builder import build_circuit
    from quipper_distributor.parsing.quipper_ascii import parse_circuit
    from quipper_distributor.partitioner import partitioner
    from quipper_distributor.preparation import prepare_circuit

    text = (CIRCUITS_DIR / "qft20").read_text()
    circuit = parse_circuit(text)
    gates = prepare_circuit(circuit.gates, keep_ccz=False)
    n_qubits = len(circuit.inputs)

    segments = partitioner(
        k=3,
        init_seg_size=7,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
        n_qubits=n_qubits,
        n_wires=n_qubits,
        gates=gates,
    )

    result_gates, new_wires, n_ebits, n_teleports = build_circuit(segments, n_qubits, n_qubits)

    assert len(result_gates) > 0
    assert n_ebits >= 0
    assert n_teleports >= 0


@pytest.mark.integration
def test_output_format():
    """End-to-end: verify output statistics are non-negative integers."""
    from quipper_distributor.dist_circuit_builder import build_circuit
    from quipper_distributor.models.gate import is_cz
    from quipper_distributor.parsing.quipper_ascii import parse_circuit
    from quipper_distributor.partitioner import partitioner
    from quipper_distributor.preparation import prepare_circuit

    text = (CIRCUITS_DIR / "simple").read_text()
    circuit = parse_circuit(text)
    gates = prepare_circuit(circuit.gates, keep_ccz=False)
    n_qubits = len(circuit.inputs)

    cz_count = sum(1 for g in gates if is_cz(g))
    assert cz_count > 0, "Expected some CZ gates after preparation"

    segments = partitioner(
        k=2,
        init_seg_size=1000,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
        n_qubits=n_qubits,
        n_wires=n_qubits,
        gates=gates,
    )

    _, _, n_ebits, n_teleports = build_circuit(segments, n_qubits, n_qubits)
    assert isinstance(n_ebits, int) and n_ebits >= 0
    assert isinstance(n_teleports, int) and n_teleports >= 0
    print(f"n_ebits={n_ebits}, n_teleports={n_teleports}, total={n_ebits+n_teleports}")
