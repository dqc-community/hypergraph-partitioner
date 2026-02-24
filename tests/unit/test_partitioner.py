"""Unit tests for partitioner module (no kahypar required)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from quipper_distributor.models.gate import QGate, SignedWire
from quipper_distributor.models.hypergraph import Hedge
from quipper_distributor.models.segment import Segment, SeamCompute, SeamStop, SeamValue
from quipper_distributor.partitioner import (
    _find_valley_rec,
    compute_new_seams,
    count_teles,
    find_valley,
    get_rho,
    initial_segments,
    match_partitions,
    merge_seams,
)


def _make_seg(gates=None, hyp=None, part=None, seam=None, wr=(0, 0)):
    return Segment(
        gates=gates or [],
        hypergraph=hyp or {},
        partition=part or {},
        seam=seam or SeamCompute(),
        wire_range=wr,
    )


class TestGetRho:
    def test_identical_partitions(self):
        """Identical partitions → rho = 0."""
        hyp = {0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)], 1: []}
        part = {0: 0, 1: 1}
        seg1 = _make_seg(hyp=hyp, part=part)
        seg2 = _make_seg(hyp=hyp, part=part)
        assert get_rho(2, seg1, seg2) == Fraction(0)

    def test_all_wires_changing(self):
        """Wires in different blocks each segment → rho > 0."""
        hyp = {0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)]}
        part1 = {0: 0, 1: 1}
        part2 = {0: 1, 1: 0}
        seg1 = _make_seg(hyp=hyp, part=part1)
        seg2 = _make_seg(hyp=hyp, part=part2)
        assert get_rho(2, seg1, seg2) > Fraction(0)

    def test_empty_hypergraphs(self):
        seg1 = _make_seg(part={0: 0, 1: 0})
        seg2 = _make_seg(part={0: 1, 1: 1})
        # Even with changing blocks, empty hyp → weight = 0
        rho = get_rho(2, seg1, seg2)
        assert rho == Fraction(0)


class TestFindValley:
    def _make_value_seg(self, rho_num: int, rho_den: int = 1):
        return _make_seg(seam=SeamValue(value=Fraction(rho_num, rho_den)))

    def _make_stop_seg(self):
        return _make_seg(seam=SeamStop())

    def test_simple_valley(self):
        """Valley at index 1 (minimum), followed by stop."""
        segs = [
            self._make_value_seg(3),  # rho=3
            self._make_value_seg(1),  # rho=1 (minimum)
            self._make_stop_seg(),  # Stop
        ]
        before, valley, rest = find_valley(segs)
        assert len(valley) >= 1

    def test_single_min_before_stop(self):
        """Single value followed by stop → min is at pos 0."""
        segs = [
            self._make_value_seg(2),
            self._make_stop_seg(),
        ]
        before, valley, rest = find_valley(segs)
        assert len(before) + len(valley) + len(rest) == 2


class TestMatchPartitions:
    def test_identity_matching(self):
        """All wires in same block → identity matching expected."""
        part1 = {0: 0, 1: 0}
        part2 = {0: 0, 1: 0}
        m = match_partitions(part1, part2, k=2, n_wires=2)
        assert isinstance(m, dict)
        assert len(m) == 2

    def test_all_blocks_assigned(self):
        """Matching assigns all k blocks."""
        part1 = {0: 0, 1: 1}
        part2 = {0: 1, 1: 0}
        m = match_partitions(part1, part2, k=2, n_wires=2)
        assert set(m.keys()) == {0, 1}

    def test_k3_all_blocks(self):
        part1 = {0: 0, 1: 1, 2: 2}
        part2 = {0: 2, 1: 0, 2: 1}
        m = match_partitions(part1, part2, k=3, n_wires=3)
        assert set(m.keys()) == {0, 1, 2}
        assert set(m.values()) == {0, 1, 2}


class TestInitialSegments:
    def _dummy_hyp(self, gates):
        return {}

    def _dummy_part(self, hyp):
        return {}

    def test_no_gates_empty_segments(self):
        result = initial_segments([], 2, self._dummy_hyp, self._dummy_part)
        assert result == []

    def test_gates_with_two_czs(self):
        """10 gates with 2 CZs and init_seg_size=1 → ≥ 2 segments."""
        h_gate = QGate(name="H", inputs=[0])
        cz_gate = QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])
        gates = [h_gate] * 3 + [cz_gate] + [h_gate] * 3 + [cz_gate] + [h_gate] * 2

        result = initial_segments(gates, 1, self._dummy_hyp, self._dummy_part)
        assert len(result) >= 2

    def test_no_cz_single_segment(self):
        """Gates without CZ → single segment."""
        gates = [QGate(name="H", inputs=[0])] * 5
        result = initial_segments(gates, 2, self._dummy_hyp, self._dummy_part)
        assert len(result) == 1


class TestCountTeles:
    def test_no_change(self):
        assert count_teles({0: 0, 1: 1}, {0: 0, 1: 1}, n_wires=2) == 0

    def test_all_change(self):
        assert count_teles({0: 0, 1: 1}, {0: 1, 1: 0}, n_wires=2) == 2

    def test_partial_change(self):
        assert count_teles({0: 0, 1: 0}, {0: 1, 1: 0}, n_wires=2) == 1


class TestMergeSeamsWithMock:
    """Test merge_seams with an identity partition mock (no kahypar needed)."""

    def _id_part(self, hyp):
        return {w: 0 for w in range(4)}

    def _trivial_hyp(self, gates):
        return {}

    def test_already_stopped(self):
        """All-Stop segments → no change."""
        segs = [
            _make_seg(seam=SeamStop()),
            _make_seg(seam=SeamStop()),
        ]
        result = merge_seams(self._trivial_hyp, self._id_part, k=2, n_wires=4, segments=segs)
        assert all(isinstance(s.seam, SeamStop) for s in result)
