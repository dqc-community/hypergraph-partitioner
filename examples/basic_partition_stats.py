"""Minimal library usage example for hypergraph-partitioner."""

from __future__ import annotations

from pathlib import Path

from bosonic_model.qasm import Translator

from hypergraph_partitioner import (
    count_interactions,
    count_nonlocal_interactions,
    count_teleports,
    partition_circuit,
)

qasm_text = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
cx q[1], q[3];
cx q[1], q[2];
cx q[1], q[0];
"""

config_path = Path(__file__).resolve().parent.parent / "kahypar/config/km1_kKaHyPar_sea20.ini"

circuit = Translator().from_qasm(qasm_text)
segments = partition_circuit(
    circuit,
    k=2,
    init_seg_size=1000,
    max_hedge_dist=100,
    config_path=str(config_path),
)

stats = {
    "interactions": count_interactions(circuit.instructions),
    "nonlocal": count_nonlocal_interactions(segments),
    "teleports": count_teleports(segments, circuit.qubits()),
}

print(stats)
