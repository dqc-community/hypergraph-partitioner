"""Seam scoring, partition matching, and segment merging helpers."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import count
from collections.abc import Callable
from fractions import Fraction

from hypergraph_partitioner.models.hypergraph import Hypergraph, Matching, Partition
from hypergraph_partitioner.models.segment import Seam, SeamCompute, SeamStop, SeamValue, Segment

ToHyp = Callable[[list[object]], Hypergraph]
ToPart = Callable[[Hypergraph], Partition]


@dataclass(frozen=True)
class QubitSpan:
    qubit: int
    start: int
    end: int
    interaction_ids: tuple[int, ...]


def merge_seams(
    to_hyp: ToHyp, to_part: ToPart, k: int, n_qubits: int, max_hedge_dist: int, segments: list[Segment]
) -> list[Segment]:
    """Iteratively merge segments until all seams are Stop."""
    id_matching: Matching = {b: b for b in range(k)}

    while True:
        if all(isinstance(s.seam, SeamStop) for s in segments):
            return segments

        matched = _match_segments(k, n_qubits, id_matching, segments)
        seamed = _compute_new_seams(n_qubits, max_hedge_dist, matched)
        merged = _merge_min(to_hyp, to_part, n_qubits, seamed)
        segments = ignore_last_seam(merged)


def ignore_last_seam(segments: list[Segment]) -> list[Segment]:
    """The last segment has no next segment, so mark its seam as Stop."""
    if not segments:
        return segments
    result = list(segments)
    last = result[-1]
    result[-1] = Segment(
        gates=last.gates,
        hypergraph=last.hypergraph,
        partition=last.partition,
        seam=SeamStop(),
        segment_range=last.segment_range,
    )
    return result


def _match_segments(
    k: int, n_qubits: int, prev_matching: Matching, segments: list[Segment]
) -> list[Segment]:
    """Rename blocks so adjacent segments match optimally."""
    if not segments:
        return []

    result: list[Segment] = []
    seg0 = _upd_with(prev_matching, segments[0])
    result.append(seg0)

    for i in range(1, len(segments)):
        prev_seg = result[-1]
        curr_seg = segments[i]

        if isinstance(prev_seg.seam, SeamCompute) or isinstance(curr_seg.seam, SeamCompute):
            new_matching = _match_partitions(curr_seg.partition, prev_seg.partition, k, n_qubits)
        else:
            new_matching = prev_matching

        updated = _upd_with(new_matching, curr_seg)
        result.append(updated)
        prev_matching = new_matching

    return result


def _compute_new_seams(n_qubits: int, max_hedge_dist: int, segments: list[Segment]) -> list[Segment]:
    """Assign rho values to Compute seams."""
    result: list[Segment] = []
    for i, seg in enumerate(segments):
        if isinstance(seg.seam, SeamCompute) and i + 1 < len(segments):
            new_seam: Seam = SeamValue(value=_get_rho(n_qubits, seg, segments[i + 1], max_hedge_dist))
        else:
            new_seam = seg.seam
        result.append(
            Segment(
                gates=seg.gates,
                hypergraph=seg.hypergraph,
                partition=seg.partition,
                seam=new_seam,
                segment_range=seg.segment_range,
            )
        )
    return result


def _merge_min(to_hyp: ToHyp, to_part: ToPart, n_qubits: int, segments: list[Segment]) -> list[Segment]:
    """Find minima in rho sequence and try merging adjacent segments."""
    if not segments:
        return []

    head = segments[0]
    if isinstance(head.seam, SeamStop):
        return [head] + _merge_min(to_hyp, to_part, n_qubits, segments[1:])

    if not isinstance(head.seam, SeamValue):
        return segments

    before_valley, valley, after_valley = _find_valley(segments)
    if len(valley) < 2:
        marked = [
            Segment(
                gates=segment.gates,
                hypergraph=segment.hypergraph,
                partition=segment.partition,
                seam=SeamStop(),
                segment_range=segment.segment_range,
            )
            for segment in before_valley + valley
        ]
        return marked + _merge_min(to_hyp, to_part, n_qubits, after_valley)

    left_seg, right_seg = valley[0], valley[1]
    merged_gates = left_seg.gates + right_seg.gates
    merged_hyp = to_hyp(merged_gates)
    merged_part = to_part(merged_hyp)
    merged_segment_range = (left_seg.segment_range[0], right_seg.segment_range[1])
    merged_seg = Segment(
        gates=merged_gates,
        hypergraph=merged_hyp,
        partition=merged_part,
        seam=SeamCompute(),
        segment_range=merged_segment_range,
    )

    cuts_left = _count_cuts(left_seg)
    cuts_right = _count_cuts(right_seg)
    teles = _count_qubit_moves(left_seg.partition, right_seg.partition, n_qubits)
    cuts_merged = _count_cuts(merged_seg)
    separate_cost = cuts_left + cuts_right + teles

    if separate_cost < cuts_merged:
        marked = [
            Segment(
                gates=segment.gates,
                hypergraph=segment.hypergraph,
                partition=segment.partition,
                seam=SeamStop(),
                segment_range=segment.segment_range,
            )
            for segment in before_valley + valley
        ]
        return marked + _merge_min(to_hyp, to_part, n_qubits, after_valley)

    remaining_valley = valley[2:]
    segments_after_valley = remaining_valley + after_valley
    return before_valley + [merged_seg] + _merge_min(
        to_hyp,
        to_part,
        n_qubits,
        segments_after_valley,
    )


def _match_partitions(part1: Partition, part2: Partition, k: int, n_qubits: int) -> Matching:
    """Match blocks between adjacent partitions with a best-first branch-and-bound search."""
    p1 = {q: b for q, b in part1.items() if q < n_qubits}
    p2 = {q: b for q, b in part2.items() if q < n_qubits}

    blocks1: dict[int, list[int]] = {b: [] for b in range(k)}
    for qubit, block in p1.items():
        blocks1[block].append(qubit)

    heuristic = _heuristic_cost(blocks1, p2)
    all_blocks = list(range(k))
    initial_cost = sum(heuristic.values())
    initial: list[tuple[int, int, Matching]] = [(initial_cost, 0, {})]

    return _match_partitions_search(blocks1, p2, heuristic, all_blocks, initial)


def _heuristic_cost(blocks1: dict[int, list[int]], part2: Partition) -> dict[int, int]:
    result: dict[int, int] = {}
    for block, qubits in blocks1.items():
        if not qubits:
            result[block] = 0
            continue

        valid = [qubit for qubit in qubits if qubit in part2]
        if not valid:
            result[block] = 0
            continue

        from collections import Counter

        counts = Counter(part2[qubit] for qubit in valid)
        max_same = max(counts.values()) if counts else 0
        result[block] = len(qubits) - max_same
    return result


def _match_partitions_search(
    blocks1: dict[int, list[int]],
    part2: Partition,
    heuristic: dict[int, int],
    all_blocks: list[int],
    queue: list[tuple[int, int, Matching]],
) -> Matching:
    heapify(queue)
    sequence = count(start=1)

    while queue:
        best_cost, _, best_matching = heappop(queue)
        if len(best_matching) == len(all_blocks):
            return best_matching

        unmatched_blocks = [block for block in all_blocks if block not in best_matching]
        this_block = unmatched_blocks[0]
        this_qubits = blocks1[this_block]

        used_targets = set(best_matching.values())
        candidate_targets = [block for block in all_blocks if block not in used_targets]

        for block in candidate_targets:
            new_matching = dict(best_matching)
            new_matching[this_block] = block
            real_cost = sum(1 for qubit in this_qubits if qubit in part2 and part2[qubit] != block)
            new_cost = best_cost - heuristic[this_block] + real_cost
            heappush(queue, (new_cost, next(sequence), new_matching))

    return {block: block for block in all_blocks}


def _get_rho(n_qubits: int, seg1: Segment, seg2: Segment, max_hedge_dist: int) -> Fraction:
    """Compute seam cost between two adjacent segments."""
    part1 = seg1.partition
    part2 = seg2.partition
    spans1 = _split_qubit_spans(seg1, max_hedge_dist)
    spans2 = _split_qubit_spans(seg2, max_hedge_dist)

    changing = [qubit for qubit in range(n_qubits) if qubit in part1 and qubit in part2 and part1[qubit] != part2[qubit]]

    def hedges(qubit: int, spans: dict[int, list[QubitSpan]]) -> int:
        return len(spans.get(qubit, []))

    def total_hs(spans: dict[int, list[QubitSpan]]) -> int:
        return sum(len(qubit_spans) for qubit_spans in spans.values())

    def weight(qubit: int, spans: dict[int, list[QubitSpan]]) -> Fraction:
        total = total_hs(spans)
        if total == 0:
            return Fraction(0)
        return Fraction(hedges(qubit, spans), total)

    total = Fraction(0)
    for qubit in changing:
        if hedges(qubit, spans1) < hedges(qubit, spans2):
            total += weight(qubit, spans1)
        else:
            total += weight(qubit, spans2)
    return total


def _instruction_qubits(inst: object) -> list[int]:
    inner = getattr(inst, "op", inst)
    return list(getattr(inner, "qubits", []) or [])


def _derive_qubit_spans(seg: Segment) -> dict[int, list[QubitSpan]]:
    interaction_id_by_position = {
        interaction.position: interaction.interaction_id for interaction in seg.hypergraph.interactions.values()
    }
    spans: dict[int, list[QubitSpan]] = {}
    open_spans: dict[int, tuple[int, list[int]]] = {}

    for pos, inst in enumerate(seg.gates):
        qubits = _instruction_qubits(inst)
        if len(qubits) >= 2:
            interaction_id = interaction_id_by_position.get(pos)
            if interaction_id is None:
                continue
            for qubit in qubits:
                start, interaction_ids = open_spans.get(qubit, (pos, []))
                interaction_ids.append(interaction_id)
                open_spans[qubit] = (start, interaction_ids)
            continue

        if len(qubits) == 1:
            qubit = qubits[0]
            start, interaction_ids = open_spans.get(qubit, (pos, []))
            spans.setdefault(qubit, []).append(
                QubitSpan(qubit=qubit, start=start, end=pos, interaction_ids=tuple(interaction_ids))
            )
            open_spans[qubit] = (pos, [])

    end_pos = len(seg.gates)
    for qubit, (start, interaction_ids) in open_spans.items():
        spans.setdefault(qubit, []).append(
            QubitSpan(qubit=qubit, start=start, end=end_pos, interaction_ids=tuple(interaction_ids))
        )

    return {
        qubit: [span for span in qubit_spans if span.interaction_ids]
        for qubit, qubit_spans in spans.items()
        if any(span.interaction_ids for span in qubit_spans)
    }


def _split_long_spans(
    spans: list[QubitSpan], interactions: dict[int, object], max_hedge_dist: int
) -> list[QubitSpan]:
    if not spans:
        return []

    result: list[QubitSpan] = []
    queue = list(spans)

    while queue:
        span = queue.pop(0)
        if max_hedge_dist == 1:
            for interaction_id in span.interaction_ids:
                position = interactions[interaction_id].position
                result.append(
                    QubitSpan(
                        qubit=span.qubit,
                        start=position,
                        end=position + 1,
                        interaction_ids=(interaction_id,),
                    )
                )
        elif max_hedge_dist < (span.end - span.start) and len(span.interaction_ids) > 1:
            split = span.start + (span.end - span.start) // 2
            before = tuple(
                interaction_id
                for interaction_id in span.interaction_ids
                if interactions[interaction_id].position < split
            )
            after = tuple(
                interaction_id
                for interaction_id in span.interaction_ids
                if interactions[interaction_id].position >= split
            )
            if after:
                queue.insert(0, QubitSpan(qubit=span.qubit, start=split, end=span.end, interaction_ids=after))
            if before:
                queue.insert(0, QubitSpan(qubit=span.qubit, start=span.start, end=split, interaction_ids=before))
        else:
            result.append(span)

    return result


def _split_qubit_spans(seg: Segment, max_hedge_dist: int) -> dict[int, list[QubitSpan]]:
    spans = _derive_qubit_spans(seg)
    return {
        qubit: _split_long_spans(qubit_spans, seg.hypergraph.interactions, max_hedge_dist)
        for qubit, qubit_spans in spans.items()
    }


def _find_valley(segments: list[Segment]) -> tuple[list[Segment], list[Segment], list[Segment]]:
    """Find the minimum-rho valley in the seam sequence."""
    min_pos, end_pos = _find_valley_rec(segments, 0, None)
    before_min = segments[:min_pos]
    valley = segments[min_pos : end_pos + 1]
    rest = segments[end_pos + 1 :]
    return before_min, valley, rest


def _find_valley_rec(segments: list[Segment], pos: int, current_min: int | None) -> tuple[int, int]:
    if len(segments) < 2:
        return (pos, pos)

    this = segments[0]
    nxt = segments[1]

    if not isinstance(this.seam, SeamValue):
        return (pos, pos)

    rho = this.seam.value
    next_seam = nxt.seam

    if current_min is None:
        if isinstance(next_seam, SeamValue):
            if rho < next_seam.value:
                return _find_valley_rec(segments[1:], pos + 1, pos)
            return _find_valley_rec(segments[1:], pos + 1, None)
        if isinstance(next_seam, SeamStop):
            return (pos, pos + 1)
        return (pos, pos)

    if isinstance(next_seam, SeamValue):
        if rho > next_seam.value:
            return (current_min, pos)
        return _find_valley_rec(segments[1:], pos + 1, current_min)
    if isinstance(next_seam, SeamStop):
        return (current_min, pos + 1)
    return (current_min, pos)


def _upd_with(matching: Matching, seg: Segment) -> Segment:
    """Apply a partition matching (rename) to a segment."""
    new_part = {qubit: matching.get(block, block) for qubit, block in seg.partition.items()}
    return Segment(
        gates=seg.gates,
        hypergraph=seg.hypergraph,
        partition=new_part,
        seam=seg.seam,
        segment_range=seg.segment_range,
    )


def _count_qubit_moves(part1: Partition, part2: Partition, n_qubits: int) -> int:
    """Count qubits that change block between two adjacent partitions."""
    return sum(1 for qubit in range(n_qubits) if qubit in part1 and qubit in part2 and part1[qubit] != part2[qubit])


def _count_cuts(segment: Segment) -> int:
    """Count the number of hyperedge cuts induced by a segment partition."""
    total = 0
    for interaction in segment.hypergraph.interactions.values():
        blocks = {segment.partition[qubit] for qubit in interaction.qubits if qubit in segment.partition}
        total += max(0, len(blocks) - 1)
    return total
