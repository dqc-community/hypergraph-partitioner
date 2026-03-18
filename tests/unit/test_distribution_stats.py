"""Unit tests for bosonic-native distribution stats."""

from __future__ import annotations

from bosonic_model.qasm import Translator

from hypergraph_partitioner.bosonic_pipeline import (
    _count_interactions,
    _count_nonlocal_interactions,
    _count_teleports,
    partition_circuit,
)
from hypergraph_partitioner.config import KAHYPAR_CONFIG


def test_stats_non_negative_for_cx_circuit() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        cx q[0], q[1];
        cx q[1], q[2];
        """
    )

    result = partition_circuit(
        circuit,
        k=2,
        init_seg_size=1000,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert _count_interactions(circuit.instructions) >= 2
    assert _count_nonlocal_interactions(result) >= 0
    assert _count_teleports(result) >= 0


def test_stats_handle_toffoli() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        ccx q[0], q[1], q[2];
        """
    )

    result = partition_circuit(
        circuit,
        k=2,
        init_seg_size=1000,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert _count_interactions(circuit.instructions) == 1
    assert _count_nonlocal_interactions(result) >= 0
