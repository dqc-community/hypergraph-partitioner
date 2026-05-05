"""Integration regressions for max_hedge_dist seam behavior."""

from __future__ import annotations

import pytest
from bosonic_model.qasm import Translator

from hypergraph_partitioner.bosonic_pipeline import (
    _count_nonlocal_interactions,
    _count_swaps,
    _count_teleports,
    _partition_to_partitioned_circuit,
)


def _multi_segment_regression_circuit():
    return Translator().from_qasm(
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


@pytest.mark.integration
def test_produces_multiple_segments() -> None:
    circuit = _multi_segment_regression_circuit()

    result = _partition_to_partitioned_circuit(
        circuit,
        nodes=2,
        init_seg_size=10,
        max_hedge_dist=100,
    )

    assert len(result.segments) == 5
    assert len(result.boundaries) == 4
    assert _count_nonlocal_interactions(result) == 14
    assert _count_swaps(result) == 4
    assert dict(result.segments[0].partition) == {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1}
    assert dict(result.segments[1].partition) == {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0}
    assert dict(result.segments[2].partition) == {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0}
    assert dict(result.segments[3].partition) == {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0}
    assert dict(result.segments[4].partition) == {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0}


@pytest.mark.integration
def test_max_hedge_dist_changes_segmentation() -> None:
    circuit = _multi_segment_regression_circuit()

    short = _partition_to_partitioned_circuit(
        circuit,
        nodes=2,
        init_seg_size=10,
        max_hedge_dist=1,
    )
    long = _partition_to_partitioned_circuit(
        circuit,
        nodes=2,
        init_seg_size=10,
        max_hedge_dist=100,
    )

    assert len(short.segments) == 4
    assert len(short.boundaries) == 3
    assert _count_nonlocal_interactions(short) == 15
    assert _count_teleports(short) == 0
    assert _count_swaps(short) == 4

    assert len(long.segments) == 5
    assert len(long.boundaries) == 4
    assert _count_nonlocal_interactions(long) == 14
    assert _count_teleports(long) == 0
    assert _count_swaps(long) == 4
