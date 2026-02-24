"""Unit tests for partitioner seam and matching helpers."""

from __future__ import annotations

from fractions import Fraction

from quipper_distributor.models.hypergraph import Hedge
from quipper_distributor.models.segment import SeamCompute, SeamStop, SeamValue, Segment
from quipper_distributor.partitioner import (
    _ignore_last_seam,
    compute_new_seams,
    count_teles,
    find_valley,
    match_partitions,
    merge_seams,
    partition_hypergraph,
)


def _seg(partition: dict[int, int], seam=SeamCompute()) -> Segment:
    hyp = {0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)], 1: [Hedge(nan=0, wires=[(0, 0)], out_pos=1)]}
    return Segment(gates=[], hypergraph=hyp, partition=partition, seam=seam, wire_range=(0, 0))


def test_partition_hypergraph_fallback(monkeypatch) -> None:
    monkeypatch.setenv("DISTRIBUTOR_PARTITIONER", "fallback")
    hyp = {0: [Hedge(nan=0, wires=[(1, 0)], out_pos=1)]}

    part = partition_hypergraph(hyp, n_qubits=4, k=2, config_path="unused.ini")

    assert part == {0: 0, 1: 1, 2: 0, 3: 1}


def test_compute_new_seams_sets_seam_value() -> None:
    segs = [_seg({0: 0, 1: 1}, SeamCompute()), _seg({0: 1, 1: 1}, SeamStop())]

    updated = compute_new_seams(2, segs)

    assert isinstance(updated[0].seam, SeamValue)
    assert updated[0].seam.value >= Fraction(0)


def test_find_valley_returns_nonempty_middle() -> None:
    segments = [
        _seg({0: 0, 1: 0}, SeamValue(value=Fraction(2, 3))),
        _seg({0: 0, 1: 1}, SeamValue(value=Fraction(1, 3))),
        _seg({0: 1, 1: 1}, SeamStop()),
    ]

    before, valley, rest = find_valley(segments)

    assert len(valley) >= 1
    assert len(before) + len(valley) + len(rest) == len(segments)


def test_match_partitions_returns_full_mapping() -> None:
    p1 = {0: 0, 1: 1, 2: 0, 3: 1}
    p2 = {0: 1, 1: 0, 2: 1, 3: 0}

    match = match_partitions(p1, p2, k=2, n_wires=4)

    assert set(match.keys()) == {0, 1}
    assert set(match.values()) == {0, 1}


def test_count_teles_counts_wire_moves() -> None:
    part1 = {0: 0, 1: 0, 2: 1}
    part2 = {0: 1, 1: 0, 2: 1}

    assert count_teles(part1, part2, n_wires=3) == 1


def test_merge_seams_single_segment_stops() -> None:
    segs = _ignore_last_seam([_seg({0: 0, 1: 1}, SeamCompute())])

    result = merge_seams(lambda _g: {}, lambda _h: {0: 0, 1: 1}, k=2, n_wires=2, segments=segs)

    assert len(result) == 1
    assert isinstance(result[0].seam, SeamStop)
