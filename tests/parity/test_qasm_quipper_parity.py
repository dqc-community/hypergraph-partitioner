"""Temporary parity tests: legacy Quipper path vs converted QASM fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from bosonic_model.qasm import Translator

from quipper_distributor.bosonic_adapter import circuit_to_legacy_gates
from quipper_distributor.config import KAHYPAR_CONFIG
from quipper_distributor.dist_circuit_builder import build_circuit
from quipper_distributor.models.gate import get_wires, is_cz
from quipper_distributor.parsing.quipper_ascii import parse_circuit
from quipper_distributor.partitioner import partitioner
from quipper_distributor.preparation import prepare_circuit

LEGACY_CIRCUITS_DIR = Path(__file__).parent.parent / "fixtures" / "quipper_circuits"
QASM_CIRCUITS_DIR = Path(__file__).parent.parent.parent / "circuits"


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


def _k_for_qubits(n_qubits: int) -> int:
    return 3 if n_qubits >= 12 else 2


@pytest.mark.parametrize("legacy_path", sorted(LEGACY_CIRCUITS_DIR.iterdir()), ids=lambda p: p.name)
def test_all_root_circuits_parity(monkeypatch, legacy_path: Path) -> None:
    monkeypatch.setenv("QUIPPER_DISTRIBUTOR_PARTITIONER", "fallback")

    qasm_path = QASM_CIRCUITS_DIR / legacy_path.name
    assert qasm_path.exists(), f"Missing converted QASM fixture: {qasm_path}"

    quipper_ascii = legacy_path.read_text()
    qasm_text = qasm_path.read_text()
    n_qubits = len(parse_circuit(quipper_ascii).inputs)
    k = _k_for_qubits(n_qubits)

    old_gate_count, old_cz_count, old_nonlocal, old_total_resources = _run_old(quipper_ascii, k=k)
    new_gate_count, new_cz_count, new_nonlocal, new_total_resources = _run_new(qasm_text, k=k)

    assert old_cz_count == new_cz_count
    assert old_nonlocal == new_nonlocal
    assert old_total_resources == new_total_resources
    assert abs(old_gate_count - new_gate_count) <= max(4, old_gate_count // 20)
