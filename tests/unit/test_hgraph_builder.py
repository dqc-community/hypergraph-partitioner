"""Unit tests for hypergraph helpers on bosonic instructions."""

from __future__ import annotations

from bosonic_model.qasm import Translator

from quipper_distributor.bosonic_pipeline import build_hypergraph_from_instructions
from quipper_distributor.hgraph_builder import _split_long_hedges, count_cuts, hypergraph_to_kahypar
from quipper_distributor.models.hypergraph import Hedge
from quipper_distributor.models.segment import SeamStop, Segment


def test_split_long_hedges_max_dist_one() -> None:
    hedges = [Hedge(nan=0, wires=[(7, 0), (8, 2)], out_pos=3)]

    split = _split_long_hedges(hedges, max_dist=1)

    assert len(split) == 2
    assert split[0].wires == [(7, 0)]
    assert split[0].out_pos == 1
    assert split[1].wires == [(8, 2)]
    assert split[1].out_pos == 3


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

    assert 0 in hyp and 1 in hyp
    assert len(hyp[0]) >= 1
    assert len(hyp[1]) >= 1
    assert all(h.wires for hedges in hyp.values() for h in hedges)


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
    seg = Segment(gates=circuit.instructions, hypergraph=hyp, partition={0: 0, 1: 1, -1: 0}, seam=SeamStop())

    assert count_cuts(seg) >= 1


def test_hypergraph_to_kahypar_skips_singleton_nets() -> None:
    # This shape appears after instruction-to-hypergraph conversion and used to
    # crash kahypar on macOS when singleton nets were emitted.
    hyp = {
        1: [Hedge(nan=0, wires=[(-3, 2), (-2, 1), (-1, 0)], out_pos=3)],
        0: [Hedge(nan=0, wires=[(-1, 0)], out_pos=1)],
        2: [Hedge(nan=0, wires=[(-2, 1)], out_pos=2)],
        3: [Hedge(nan=0, wires=[(-3, 2)], out_pos=3)],
    }

    indices, nets, weights = hypergraph_to_kahypar(hyp, n_qubits=4)

    assert indices == [0]
    assert nets == []
    assert weights == [1, 1, 1, 1]
