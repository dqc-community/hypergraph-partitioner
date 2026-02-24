"""Unit tests for distributed circuit builder."""

from __future__ import annotations

import pytest

from quipper_distributor.dist_circuit_builder import (
    EbitDisentangler,
    EbitEntangler,
    _entangler_gates,
    build_circuit,
    distribute_gates,
    ebit_info,
    non_local_connections,
)
from quipper_distributor.models.gate import QGate, SignedWire
from quipper_distributor.models.hypergraph import Hedge
from quipper_distributor.models.segment import Segment, SeamStop


def _make_seg(gates=None, hyp=None, part=None):
    return Segment(
        gates=gates or [],
        hypergraph=hyp or {},
        partition=part or {},
        seam=SeamStop(),
        wire_range=(0, 0),
    )


class TestNonLocalConnections:
    def test_same_block_no_connections(self):
        """Wires in same block → no non-local connections."""
        hyp = {0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)]}
        part = {0: 0, 1: 0}
        result = non_local_connections(part, hyp)
        assert result == []

    def test_different_blocks_one_connection(self):
        """Wires in different blocks → 1 non-local connection."""
        hyp = {0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)]}
        part = {0: 0, 1: 1}
        result = non_local_connections(part, hyp)
        assert len(result) == 1
        init_pos, src, snk, cz_pos, out_pos = result[0]
        assert src == 0
        assert snk == 1


class TestEbitInfo:
    def test_one_nonlocal_yields_two_components(self):
        """1 non-local connection → 1 Entangler + 1 Disentangler."""
        nonlocal_cs = [(0, 0, 1, 0, 1)]  # (init, ctrl, snk, cz_pos, out)
        part = {0: 0, 1: 1}
        components = ebit_info(part, nonlocal_cs)
        assert len(components) == 2
        entanglers = [c for c in components if isinstance(c, EbitEntangler)]
        disentanglers = [c for c in components if isinstance(c, EbitDisentangler)]
        assert len(entanglers) == 1
        assert len(disentanglers) == 1

    def test_deduplication(self):
        """Duplicate connections produce only one Entangler/Disentangler pair."""
        nonlocal_cs = [
            (0, 0, 1, 0, 1),
            (0, 0, 1, 0, 1),  # duplicate
        ]
        part = {0: 0, 1: 1}
        components = ebit_info(part, nonlocal_cs)
        assert len(components) == 2


class TestBuildCircuit:
    def test_identity_partition_no_teleports(self):
        """Two segments with identical partitions → no teleport gates."""
        cz = QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])
        seg1 = _make_seg(
            gates=[cz],
            hyp={0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)]},
            part={0: 0, 1: 0},
        )
        seg2 = _make_seg(
            gates=[QGate(name="H", inputs=[0])],
            hyp={},
            part={0: 0, 1: 0},
        )
        gates, new_wires, n_ebits, n_tps = build_circuit([seg1, seg2], n_wires=2, n_inputs=2)
        assert n_tps == 0

    def test_wires_changing_blocks_produce_teleports(self):
        """When a wire changes block between segments, a teleport gate is inserted."""
        seg1 = _make_seg(
            gates=[QGate(name="H", inputs=[0])],
            hyp={},
            part={0: 0, 1: 0},
        )
        seg2 = _make_seg(
            gates=[QGate(name="H", inputs=[0])],
            hyp={},
            part={0: 1, 1: 0},  # wire 0 changes block
        )
        gates, new_wires, n_ebits, n_tps = build_circuit([seg1, seg2], n_wires=2, n_inputs=2)
        assert n_tps >= 1
        tele_gates = [g for g in gates if isinstance(g, QGate) and g.name == "teleport"]
        assert len(tele_gates) >= 1

    def test_single_segment_no_teleports(self):
        """Single segment → no teleport gates (n_teleports = 0)."""
        seg = _make_seg(
            gates=[QGate(name="H", inputs=[0])],
            hyp={},
            part={0: 0},
        )
        gates, new_wires, n_ebits, n_tps = build_circuit([seg], n_wires=1, n_inputs=1)
        assert n_tps == 0

    def test_empty_segments(self):
        gates, new_wires, n_ebits, n_tps = build_circuit([], n_wires=2, n_inputs=2)
        assert gates == []
        assert n_ebits == 0
        assert n_tps == 0


class TestEntanglerGateSequence:
    def test_entangler_gate_count(self):
        """Entangler sequence should produce the expected number of gates."""
        from quipper_distributor.dist_circuit_builder import _entangler_gates
        gates = _entangler_gates(source=0, source_e=-1, sink_e=-2, b_sink=1, b_source=0)
        # bell(4) + not(1) + meas(1) + X(1) + comment(1) + cdiscard(1) = 9
        assert len(gates) == 9

    def test_non_local_cz_ebit_count(self):
        """1 non-local CZ → 1 ebit (= 1 entangler + 1 disentangler)."""
        cz = QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])
        seg = _make_seg(
            gates=[cz],
            hyp={0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)]},
            part={0: 0, 1: 1},
        )
        gates, new_wires, n_ebits = distribute_gates(seg.gates, seg.hypergraph, seg.partition)
        assert n_ebits == 1
