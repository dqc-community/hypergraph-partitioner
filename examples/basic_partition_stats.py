"""Minimal library usage example for hypergraph-partitioner."""

from __future__ import annotations

import os

from bosonic_model.qasm import Translator

from hypergraph_partitioner.bosonic_pipeline import (
    _partition_to_partitioned_circuit,
    _count_interactions,
    _count_nonlocal_interactions,
    _count_teleports,
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

circuit = Translator().from_qasm(qasm_text)
partitioned_circuit = _partition_to_partitioned_circuit(
    circuit,
    nodes=2,
    init_seg_size=int(os.environ.get("INIT_SEG_SIZE", "1000")),
    max_hedge_dist=100,
)


stats = {
    "interactions": _count_interactions(circuit.instructions),
    "nonlocal": _count_nonlocal_interactions(partitioned_circuit),
    "teleports": _count_teleports(partitioned_circuit),
}

print(stats)
