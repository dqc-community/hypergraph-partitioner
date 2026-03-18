"""Unit tests for partitioner seam and matching helpers."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from hypergraph_partitioner.config import KAHYPAR_CONFIG
from hypergraph_partitioner.hgraph_builder import count_cuts
from hypergraph_partitioner.models.hypergraph import Hypergraph, InteractionVertex, QubitVertex
from hypergraph_partitioner.models.segment import SeamCompute, SeamStop, SeamValue, Segment
from hypergraph_partitioner.partitioner import (
    QubitSpan,
    _find_valley_rec,
    _heuristic_cost,
    _ignore_last_seam,
    _match_partitions_search,
    _derive_qubit_spans,
    _split_long_spans,
    _upd_with,
    compute_new_seams,
    count_qubit_moves,
    find_valley,
    get_rho,
    match_partitions,
    match_segments,
    merge_min,
    merge_seams,
    partition_hypergraph,
)


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


def _gate(*qubits: int) -> SimpleNamespace:
    return SimpleNamespace(qubits=list(qubits))


def _seg(partition: dict[int, int], seam: SeamCompute | SeamStop | SeamValue = SeamCompute()) -> Segment:
    hyp = _hyp((0, 0, (0, 1)))
    return Segment(gates=[_gate(0, 1)], hypergraph=hyp, partition=partition, seam=seam, segment_range=(0, 0))


def _seg_with_hyp(
    partition: dict[int, int],
    hyp: Hypergraph,
    seam: SeamCompute | SeamStop | SeamValue = SeamCompute(),
) -> Segment:
    return Segment(gates=[], hypergraph=hyp, partition=partition, seam=seam, segment_range=(0, 0))


def test_partition_hypergraph_empty_hypergraph_returns_single_block() -> None:
    part = partition_hypergraph(Hypergraph(qubits={}, interactions={}), n_qubits=3, k=2, config_path=KAHYPAR_CONFIG)

    assert part == {0: 0, 1: 0, 2: 0}


def test_partition_hypergraph_returns_assignment_for_nonempty_hypergraph() -> None:
    hyp = _hyp((0, 0, (0, 1)))

    part = partition_hypergraph(hyp, n_qubits=2, k=2, config_path=KAHYPAR_CONFIG)

    assert set(part.keys()) == {0, 1}
    assert all(block in {0, 1} for block in part.values())


def test_ignore_last_seam_marks_last_segment_stop() -> None:
    segs = [_seg({0: 0, 1: 1}, SeamCompute()), _seg({0: 1, 1: 0}, SeamCompute())]

    updated = _ignore_last_seam(segs)

    assert isinstance(updated[-1].seam, SeamStop)
    assert updated[0].seam == segs[0].seam


def test_upd_with_renames_partition_blocks_only() -> None:
    seg = _seg({0: 0, 1: 1}, SeamValue(value=Fraction(1, 2)))

    updated = _upd_with({0: 1, 1: 0}, seg)

    assert updated.partition == {0: 1, 1: 0}
    assert updated.hypergraph == seg.hypergraph
    assert updated.gates == seg.gates
    assert updated.seam == seg.seam


def test_compute_new_seams_sets_seam_value() -> None:
    segs = [_seg({0: 0, 1: 1}, SeamCompute()), _seg({0: 1, 1: 1}, SeamStop())]

    updated = compute_new_seams(2, 100, segs)

    assert isinstance(updated[0].seam, SeamValue)
    assert updated[0].seam.value >= Fraction(0)


def test_get_rho_is_zero_when_no_qubits_change_blocks() -> None:
    seg1 = _seg({0: 0, 1: 1})
    seg2 = _seg({0: 0, 1: 1})

    rho = get_rho(2, seg1, seg2, 100)

    assert rho == Fraction(0)


def test_get_rho_uses_smaller_of_adjacent_qubit_weights() -> None:
    hyp1 = _hyp((0, 0, (0, 1)))
    hyp2 = _hyp((0, 0, (0, 1)), (1, 2, (0, 1)))
    seg1 = Segment(
        gates=[_gate(0, 1)],
        hypergraph=hyp1,
        partition={0: 0, 1: 1},
        seam=SeamCompute(),
        segment_range=(0, 0),
    )
    seg2 = Segment(
        gates=[_gate(0, 1), _gate(0), _gate(0, 1)],
        hypergraph=hyp2,
        partition={0: 1, 1: 1},
        seam=SeamCompute(),
        segment_range=(0, 0),
    )

    rho = get_rho(2, seg1, seg2, 100)

    assert rho == Fraction(1, 2)


def test_derive_qubit_spans_splits_on_single_qubit_boundaries() -> None:
    seg = Segment(
        gates=[_gate(0, 1), _gate(0), _gate(0, 1)],
        hypergraph=_hyp((0, 0, (0, 1)), (1, 2, (0, 1))),
        partition={0: 0, 1: 1},
        seam=SeamCompute(),
        segment_range=(0, 0),
    )

    spans = _derive_qubit_spans(seg)

    assert spans[0] == [
        QubitSpan(qubit=0, start=0, end=1, interaction_ids=(0,)),
        QubitSpan(qubit=0, start=1, end=3, interaction_ids=(1,)),
    ]
    assert spans[1] == [QubitSpan(qubit=1, start=0, end=3, interaction_ids=(0, 1))]


def test_derive_qubit_spans_starts_at_first_actual_qubit_use() -> None:
    seg = Segment(
        gates=[_gate(0), _gate(1), _gate(1, 2)],
        hypergraph=_hyp((0, 2, (1, 2)), qubits=(0, 1, 2)),
        partition={0: 0, 1: 0, 2: 1},
        seam=SeamCompute(),
        segment_range=(0, 0),
    )

    spans = _derive_qubit_spans(seg)

    assert spans[1] == [QubitSpan(qubit=1, start=1, end=3, interaction_ids=(0,))]
    assert spans[2] == [QubitSpan(qubit=2, start=2, end=3, interaction_ids=(0,))]


def test_split_long_spans_max_dist_one_splits_per_interaction() -> None:
    interactions = {
        0: InteractionVertex(interaction_id=0, position=0, qubits=(0, 1)),
        1: InteractionVertex(interaction_id=1, position=2, qubits=(0, 1)),
    }
    spans = [QubitSpan(qubit=0, start=0, end=3, interaction_ids=(0, 1))]

    split = _split_long_spans(spans, interactions, max_hedge_dist=1)

    assert split == [
        QubitSpan(qubit=0, start=0, end=1, interaction_ids=(0,)),
        QubitSpan(qubit=0, start=2, end=3, interaction_ids=(1,)),
    ]


def test_get_rho_changes_when_max_hedge_dist_changes() -> None:
    seg1 = Segment(
        gates=[_gate(0, 1), _gate(0, 1), _gate(2, 3)],
        hypergraph=_hyp((0, 0, (0, 1)), (1, 1, (0, 1)), (2, 2, (2, 3)), qubits=(0, 1, 2, 3)),
        partition={0: 0, 1: 1, 2: 0, 3: 1},
        seam=SeamCompute(),
        segment_range=(0, 0),
    )
    seg2 = Segment(
        gates=[_gate(0, 1), _gate(0), _gate(0, 1), _gate(2, 3)],
        hypergraph=_hyp((0, 0, (0, 1)), (1, 2, (0, 1)), (2, 3, (2, 3)), qubits=(0, 1, 2, 3)),
        partition={0: 1, 1: 1, 2: 0, 3: 1},
        seam=SeamCompute(),
        segment_range=(0, 0),
    )

    assert get_rho(4, seg1, seg2, 100) == Fraction(1, 4)
    assert get_rho(4, seg1, seg2, 1) == Fraction(1, 3)


def test_find_valley_returns_nonempty_middle() -> None:
    segments = [
        _seg({0: 0, 1: 0}, SeamValue(value=Fraction(2, 3))),
        _seg({0: 0, 1: 1}, SeamValue(value=Fraction(1, 3))),
        _seg({0: 1, 1: 1}, SeamStop()),
    ]

    before, valley, rest = find_valley(segments)

    assert len(valley) >= 1
    assert len(before) + len(valley) + len(rest) == len(segments)


def test_find_valley_rec_returns_valley_bounds() -> None:
    segments = [
        _seg({0: 0, 1: 0}, SeamValue(value=Fraction(3, 4))),
        _seg({0: 0, 1: 1}, SeamValue(value=Fraction(1, 4))),
        _seg({0: 1, 1: 1}, SeamStop()),
    ]

    start, end = _find_valley_rec(segments, 0, None)

    assert (start, end) == (1, 2)


def test_heuristic_cost_counts_best_block_overlap_per_source_block() -> None:
    blocks1 = {0: [0, 2], 1: [1, 3], 2: []}
    part2 = {0: 1, 1: 0, 2: 1, 3: 0}

    heuristic = _heuristic_cost(blocks1, part2)

    assert heuristic == {0: 0, 1: 0, 2: 0}


def test_match_partitions_returns_full_mapping() -> None:
    p1 = {0: 0, 1: 1, 2: 0, 3: 1}
    p2 = {0: 1, 1: 0, 2: 1, 3: 0}

    match = match_partitions(p1, p2, k=2, n_qubits=4)

    assert set(match.keys()) == {0, 1}
    assert set(match.values()) == {0, 1}
    assert match == {0: 1, 1: 0}


def test_match_partitions_prefers_lowest_cost_full_matching() -> None:
    p1 = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    p2 = {0: 2, 1: 2, 2: 0, 3: 0, 4: 1, 5: 1}

    match = match_partitions(p1, p2, k=3, n_qubits=6)

    assert match == {0: 2, 1: 0, 2: 1}


def test_match_partitions_handles_nonzero_optimal_cost() -> None:
    p1 = {0: 0, 1: 0, 2: 1, 3: 1}
    p2 = {0: 0, 1: 1, 2: 0, 3: 1}

    match = match_partitions(p1, p2, k=2, n_qubits=4)

    assert match == {0: 0, 1: 1}


def test_match_partitions_handles_empty_blocks() -> None:
    p1 = {0: 0, 1: 0}
    p2 = {0: 1, 1: 1}

    match = match_partitions(p1, p2, k=3, n_qubits=2)

    assert match == {0: 1, 1: 0, 2: 2}


def test_match_partitions_preserves_identity_when_already_aligned() -> None:
    p1 = {0: 0, 1: 1}
    p2 = {0: 0, 1: 1}

    match = match_partitions(p1, p2, k=3, n_qubits=2)

    assert match == {0: 0, 1: 1, 2: 2}


def test_match_partitions_search_returns_best_full_matching() -> None:
    blocks1 = {0: [0, 2], 1: [1, 3]}
    part2 = {0: 1, 1: 0, 2: 1, 3: 0}
    heuristic = {0: 0, 1: 0}

    match = _match_partitions_search(
        blocks1,
        part2,
        heuristic,
        [0, 1],
        [(0, 0, {})],
    )

    assert match == {0: 1, 1: 0}


def test_match_segments_aligns_adjacent_block_labels() -> None:
    segments = [
        _seg({0: 0, 1: 1}, SeamCompute()),
        _seg({0: 1, 1: 0}, SeamCompute()),
    ]

    matched = match_segments(2, 2, {0: 0, 1: 1}, segments)

    assert matched[0].partition == {0: 0, 1: 1}
    assert matched[1].partition == {0: 0, 1: 1}


def test_count_qubit_moves_counts_qubit_moves() -> None:
    part1 = {0: 0, 1: 0, 2: 1}
    part2 = {0: 1, 1: 0, 2: 1}

    assert count_qubit_moves(part1, part2, n_qubits=3) == 1


def test_count_cuts_counts_cross_block_hyperedges() -> None:
    seg = Segment(
        gates=[],
        hypergraph=_hyp((0, 0, (0, 1)), (1, 1, (0, 1))),
        partition={0: 0, 1: 1},
        seam=SeamCompute(),
        segment_range=(0, 0),
    )

    assert count_cuts(seg) == 2


def test_merge_min_merges_when_merged_cut_cost_is_not_worse() -> None:
    left = _seg({0: 0, 1: 1}, SeamValue(value=Fraction(1, 4)))
    right = _seg({0: 1, 1: 0}, SeamStop())

    result = merge_min(
        lambda _insts: _hyp((0, 0, (0,))),
        lambda _hyp: {0: 0, 1: 0},
        n_qubits=2,
        segments=[left, right],
    )

    assert len(result) == 1
    assert result[0].partition == {0: 0, 1: 0}
    assert isinstance(result[0].seam, SeamCompute)


def test_merge_seams_single_segment_stops() -> None:
    segs = _ignore_last_seam([_seg({0: 0, 1: 1}, SeamCompute())])

    result = merge_seams(
        lambda _g: Hypergraph(qubits={}, interactions={}),
        lambda _h: {0: 0, 1: 1},
        k=2,
        n_qubits=2,
        max_hedge_dist=100,
        segments=segs,
    )

    assert len(result) == 1
    assert isinstance(result[0].seam, SeamStop)
