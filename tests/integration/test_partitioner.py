from __future__ import annotations

import pytest

from hypergraph_partitioner.config import KAHYPAR_CONFIG
from hypergraph_partitioner.kahypar_partitioner import partition_hypergraph
from hypergraph_partitioner.models.hypergraph import Hypergraph, InteractionVertex, QubitVertex


def _hyp(*interactions: tuple[int, int, tuple[int, ...]], qubits: tuple[int, ...] = (0, 1)) -> Hypergraph:
    return Hypergraph(
        qubits={qubit_id: QubitVertex(qubit_id) for qubit_id in qubits},
        interactions={
            interaction_id: InteractionVertex(
                interaction_id=interaction_id,
                position=position,
                qubits=qubits,
            )
            for interaction_id, position, qubits in interactions
        },
    )


@pytest.mark.integration
def test_partition_hypergraph_empty_hypergraph_returns_balanced_assignment() -> None:
    part = partition_hypergraph(Hypergraph(qubits={}, interactions={}), n_qubits=3, nodes=2, config_path=KAHYPAR_CONFIG)

    assert part == {0: 0, 1: 1, 2: 0}


@pytest.mark.integration
def test_partition_hypergraph_returns_assignment_for_nonempty_hypergraph() -> None:
    hyp = _hyp((0, 0, (0, 1)))

    part = partition_hypergraph(hyp, n_qubits=2, nodes=2, config_path=KAHYPAR_CONFIG)

    assert part.keys() == {0, 1}
    assert sorted(part.values()) == [0, 1]
