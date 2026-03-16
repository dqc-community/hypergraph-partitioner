"""Integration tests for QASM + bosonic_model runtime pipeline."""

from __future__ import annotations

import pytest
from bosonic_model.qasm import Translator

from hypergraph_partitioner.bosonic_pipeline import (
    count_interactions,
    count_nonlocal_interactions,
    count_teleports,
    partition_circuit,
)
from hypergraph_partitioner.config import KAHYPAR_CONFIG


def _run_pipeline(qasm_text: str, k: int, init_seg_size: int = 1000) -> tuple[int, int, int]:
    circuit = Translator().from_qasm(qasm_text)

    result = partition_circuit(
        circuit,
        k=k,
        init_seg_size=init_seg_size,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    interaction_count = count_interactions(circuit.instructions)
    nonlocal_count = count_nonlocal_interactions(result)
    teleports = count_teleports(result)
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

    interaction_count, nonlocal_count, teleports = _run_pipeline(qasm, k=2)

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

    interaction_count, nonlocal_count, teleports = _run_pipeline(qasm, k=2)

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

    _, nonlocal_count, teleports = _run_pipeline(qasm, k=2)

    assert isinstance(nonlocal_count, int) and nonlocal_count >= 0
    assert isinstance(teleports, int) and teleports >= 0


@pytest.mark.integration
def test_produces_multiple_segments() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[8];
        cx q[6],q[0];
        s q[7];
        cx q[4],q[3];
        t q[3];
        swap q[2],q[7];
        y q[1];
        ccx q[4],q[7],q[5];
        ccx q[2],q[7],q[0];
        x q[5];
        cz q[1],q[2];
        cx q[5],q[4];
        z q[7];
        cz q[4],q[0];
        swap q[0],q[7];
        cx q[0],q[4];
        cz q[5],q[1];
        t q[1];
        z q[3];
        z q[2];
        swap q[7],q[0];
        x q[5];
        swap q[7],q[0];
        s q[4];
        x q[5];
        swap q[3],q[6];
        ccx q[4],q[3],q[0];
        ccx q[6],q[2],q[4];
        z q[4];
        y q[3];
        y q[0];
        ccx q[4],q[3],q[0];
        x q[2];
        y q[0];
        x q[6];
        swap q[4],q[7];
        z q[3];
        ccx q[6],q[4],q[2];
        cz q[7],q[5];
        t q[1];
        t q[1];
        cz q[5],q[6];
        z q[3];
        h q[4];
        x q[3];
        t q[2];
        t q[6];
        h q[1];
        y q[3];
        h q[1];
        h q[1];
        z q[1];
        cx q[1],q[2];
        x q[0];
        ccx q[0],q[1],q[6];
        x q[7];
        z q[0];
        h q[6];
        ccx q[1],q[6],q[2];
        x q[3];
        x q[4];
        """
    )

    result = partition_circuit(
        circuit,
        k=2,
        init_seg_size=10,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert len(result.segments) == 4
    assert len(result.boundaries) == 3
    assert count_nonlocal_interactions(result) == 15
    assert count_teleports(result) == 8
    assert dict(result.segments[0].partition) == {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1}
    assert dict(result.segments[1].partition) == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0}
    assert dict(result.segments[2].partition) == {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0}
    assert dict(result.segments[3].partition) == {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0}
