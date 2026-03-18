"""Minimal library usage example for hypergraph-partitioner."""

from __future__ import annotations

import os
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
qreg q[6];
cx q[0], q[1];
cx q[2], q[3];
cx q[4], q[5];
cx q[0], q[3];
cx q[1], q[4];
"""

config_path = (
    Path(__file__).resolve().parent.parent
    / "src/hypergraph_partitioner/kahypar_partioner/config/km1_kKaHyPar_sea20.ini"
)

circuit = Translator().from_qasm(qasm_text)
partitioned_circuit = partition_circuit(
    circuit,
    k=2,
    init_seg_size=int(os.environ.get("INIT_SEG_SIZE", "1000")),
    max_hedge_dist=100,
    config_path=str(config_path),
)


stats = {
    "interactions": count_interactions(circuit.instructions),
    "nonlocal": count_nonlocal_interactions(partitioned_circuit),
    "teleports": count_teleports(partitioned_circuit),
}

print(stats)
