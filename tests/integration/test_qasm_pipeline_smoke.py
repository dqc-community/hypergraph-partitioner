"""Integration tests for QASM + bosonic_model runtime pipeline."""

from __future__ import annotations

import pytest
from bosonic_model.qasm import Translator

from hypergraph_partitioner.bosonic_pipeline import (
    _count_interactions,
    _count_nonlocal_interactions,
    _count_teleports,
    _partition_to_partitioned_circuit,
)


def _run_pipeline(qasm_text: str, nodes: int, init_seg_size: int = 1000) -> tuple[int, int, int]:
    circuit = Translator().from_qasm(qasm_text)

    result = _partition_to_partitioned_circuit(
        circuit,
        nodes=nodes,
        init_seg_size=init_seg_size,
        max_hedge_dist=100,
    )

    interaction_count = _count_interactions(circuit.instructions)
    nonlocal_count = _count_nonlocal_interactions(result)
    teleports = _count_teleports(result)
    return interaction_count, nonlocal_count, teleports


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

    interaction_count, nonlocal_count, teleports = _run_pipeline(qasm, nodes=2)

    assert interaction_count > 0
    assert nonlocal_count >= 0
    assert teleports >= 0


@pytest.mark.integration
def test_toffoli_qasm_k2() -> None:
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[3];

    ccx q[0], q[1], q[2];
    """

    interaction_count, nonlocal_count, teleports = _run_pipeline(qasm, nodes=2)

    assert interaction_count > 0
    assert nonlocal_count >= 0
    assert teleports >= 0


@pytest.mark.integration
def test_output_stats_non_negative() -> None:
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[2];

    h q[0];
    cx q[0], q[1];
    """

    _, nonlocal_count, teleports = _run_pipeline(qasm, nodes=2)

    assert isinstance(nonlocal_count, int) and nonlocal_count >= 0
    assert isinstance(teleports, int) and teleports >= 0
