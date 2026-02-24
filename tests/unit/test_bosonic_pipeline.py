"""Unit tests for bosonic-model-native partition pipeline."""

from __future__ import annotations

from bosonic_model.qasm import Translator

from quipper_distributor.bosonic_pipeline import (
    count_interactions,
    count_nonlocal_interactions,
    count_teleports,
    partition_circuit,
)
from quipper_distributor.config import KAHYPAR_CONFIG


def test_partition_circuit_runs_on_small_qasm(monkeypatch) -> None:
    monkeypatch.setenv("QUIPPER_DISTRIBUTOR_PARTITIONER", "fallback")

    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        h q[0];
        cx q[0], q[1];
        ccx q[0], q[1], q[2];
        """
    )

    segments = partition_circuit(
        circuit,
        k=2,
        init_seg_size=1000,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert len(segments) >= 1
    assert count_interactions(circuit.instructions) >= 2
    assert count_nonlocal_interactions(segments) >= 0
    assert count_teleports(segments, circuit.qubits()) >= 0
