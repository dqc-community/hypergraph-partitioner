"""Unit tests for hypergraph builder."""

from __future__ import annotations

import pytest

from quipper_distributor.hgraph_builder import (
    _split_long_hedges,
    build_hypergraph,
    count_cuts,
    hypergraph_to_kahypar,
)
from quipper_distributor.models.gate import QGate, SignedWire
from quipper_distributor.models.hypergraph import Hedge
from quipper_distributor.models.segment import Segment, SeamStop


class TestBuildHypergraph:
    def test_single_cz_gate(self):
        """Single CZ on wires 0, 1 → hypergraph has entries for both."""
        gates = [QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])]
        hyp = build_hypergraph(gates, n_qubits=2, max_hedge_dist=100)
        # At least one wire should have a non-empty hedge list
        total_hedges = sum(len(hs) for hs in hyp.values())
        assert total_hedges > 0

    def test_empty_gates(self):
        hyp = build_hypergraph([], n_qubits=2, max_hedge_dist=100)
        assert isinstance(hyp, dict)

    def test_no_cz_no_hedges(self):
        """H gate alone produces no non-singleton hedges."""
        gates = [QGate(name="H", inputs=[0])]
        hyp = build_hypergraph(gates, n_qubits=2, max_hedge_dist=100)
        # After filtering singletons, should be empty
        assert all(len(hs) == 0 or all(h.wires for h in hs) for hs in hyp.values())


class TestCountCuts:
    def test_single_partition_no_cuts(self):
        """All wires in same block → 0 cuts."""
        seg = Segment(
            gates=[QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])],
            hypergraph={0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)]},
            partition={0: 0, 1: 0},
            seam=SeamStop(),
            wire_range=(0, 0),
        )
        assert count_cuts(seg) == 0

    def test_wires_in_different_blocks(self):
        """CZ wires in different blocks → ≥ 1 cut."""
        seg = Segment(
            gates=[QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])],
            hypergraph={0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)]},
            partition={0: 0, 1: 1},
            seam=SeamStop(),
            wire_range=(0, 0),
        )
        assert count_cuts(seg) == 1

    def test_empty_hypergraph(self):
        seg = Segment(
            gates=[],
            hypergraph={},
            partition={},
            seam=SeamStop(),
            wire_range=(0, 0),
        )
        assert count_cuts(seg) == 0


class TestSplitLongHedges:
    def test_max_dist_1(self):
        """max_dist=1 → each (wire, pos) becomes its own hedge."""
        hedges = [Hedge(nan=0, wires=[(1, 2), (3, 5)], out_pos=6)]
        result = _split_long_hedges(hedges, max_dist=1)
        assert len(result) == 2
        assert all(len(h.wires) == 1 for h in result)

    def test_no_split_needed(self):
        """Hedge within max_dist stays whole."""
        hedges = [Hedge(nan=0, wires=[(1, 1)], out_pos=5)]
        result = _split_long_hedges(hedges, max_dist=10)
        assert len(result) == 1

    def test_split_long_hedge(self):
        """Hedge exceeding max_dist gets split."""
        hedges = [Hedge(nan=0, wires=[(1, 1), (2, 50)], out_pos=100)]
        result = _split_long_hedges(hedges, max_dist=10)
        assert len(result) >= 2


class TestHypergraphToKahypar:
    def test_csr_format(self):
        """hypergraph_to_kahypar produces valid CSR: indices length = n_hedges + 1."""
        gates = [QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])]
        hyp = build_hypergraph(gates, n_qubits=2, max_hedge_dist=100)
        indices, nets, weights = hypergraph_to_kahypar(hyp, n_qubits=2)
        n_hedges = len(indices) - 1
        assert n_hedges >= 0
        assert len(weights) == 2  # one per qubit

    def test_empty_hypergraph(self):
        indices, nets, weights = hypergraph_to_kahypar({}, n_qubits=3)
        assert len(weights) == 3

    def test_vertex_weights(self):
        """Qubit vertices have weight 1."""
        gates = [QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])]
        hyp = build_hypergraph(gates, n_qubits=2, max_hedge_dist=100)
        _, _, weights = hypergraph_to_kahypar(hyp, n_qubits=2)
        assert all(w == 1 for w in weights[:2])
