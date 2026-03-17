"""Unit tests for hypergraph helpers on bosonic instructions."""

from __future__ import annotations

from bosonic_model.qasm import Translator

from hypergraph_partitioner.bosonic_pipeline import build_hypergraph_from_instructions
from hypergraph_partitioner.hgraph_builder import (
    build_interaction_to_wires,
    count_cuts,
    hypergraph_to_kahypar,
)
from hypergraph_partitioner.models.hypergraph import Hypergraph, InteractionVertex, QubitVertex
from hypergraph_partitioner.models.segment import SeamStop, Segment


def test_build_hypergraph_from_qasm_interaction() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """
    )

    hyp = build_hypergraph_from_instructions(circuit.instructions, n_qubits=2, max_hedge_dist=100)

    assert set(hyp.wires) == {0, 1}
    assert len(hyp.interactions) >= 1
    assert any(interaction.qubits == (0, 1) for interaction in hyp.interactions.values())


def test_count_cuts_detects_cut() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        cx q[0], q[1];
        """
    )
    hyp = build_hypergraph_from_instructions(circuit.instructions, n_qubits=2, max_hedge_dist=100)
    seg = Segment(
        gates=circuit.instructions,
        hypergraph=hyp,
        partition={0: 0, 1: 1},
        seam=SeamStop(),
    )

    assert count_cuts(seg) >= 1


def test_build_interaction_to_wires_tracks_incident_real_wires() -> None:
    hyp = Hypergraph(
        qubits={0: QubitVertex(0), 1: QubitVertex(1)},
        interactions={0: InteractionVertex(interaction_id=0, position=0, qubits=(0, 1))},
    )

    assert build_interaction_to_wires(hyp) == {0: {0, 1}}


def test_hypergraph_qubit_to_interactions_orders_by_position() -> None:
    hyp = Hypergraph(
        qubits={0: QubitVertex(0), 1: QubitVertex(1), 2: QubitVertex(2), 3: QubitVertex(3)},
        interactions={
            3: InteractionVertex(interaction_id=3, position=2, qubits=(1, 3)),
            1: InteractionVertex(interaction_id=1, position=0, qubits=(0, 1)),
            2: InteractionVertex(interaction_id=2, position=1, qubits=(1, 2)),
        },
    )

    assert hyp.qubit_to_interactions == {0: [1], 1: [1, 2, 3], 2: [2], 3: [3]}


def test_hypergraph_to_kahypar_reconstructs_interaction_nets() -> None:
    hyp = Hypergraph(
        qubits={0: QubitVertex(0), 1: QubitVertex(1), 2: QubitVertex(2), 3: QubitVertex(3)},
        interactions={
            1: InteractionVertex(interaction_id=1, position=0, qubits=(1, 3)),
            2: InteractionVertex(interaction_id=2, position=1, qubits=(1, 2)),
            3: InteractionVertex(interaction_id=3, position=2, qubits=(0, 1)),
        },
    )

    indices, nets, weights = hypergraph_to_kahypar(hyp, n_qubits=4)

    assert indices == [0, 2, 4, 6]
    assert nets == [1, 3, 1, 2, 0, 1]
    assert weights == [1, 1, 1, 1]
