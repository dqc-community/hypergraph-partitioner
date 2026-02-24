"""KaHyPar bindings and segment seam/merge management."""

from __future__ import annotations

import os
import platform
from collections.abc import Callable
from fractions import Fraction

from quipper_distributor import config
from quipper_distributor.hgraph_builder import count_cuts, hypergraph_to_kahypar
from quipper_distributor.models.hypergraph import Hypergraph, Matching, Partition, Wire
from quipper_distributor.models.segment import Seam, SeamCompute, SeamStop, SeamValue, Segment

ToHyp = Callable[[list[object]], Hypergraph]
ToPart = Callable[[Hypergraph], Partition]


def partition_hypergraph(hyp: Hypergraph, n_qubits: int, k: int, config_path: str) -> Partition:
    """Partition hypergraph.

    Mode is controlled by DISTRIBUTOR_PARTITIONER:
    - "kahypar" (default): always use KaHyPar
    - "auto": use fallback on macOS, KaHyPar elsewhere
    - "fallback": always use deterministic round-robin assignment
    """
    mode = os.environ.get(
        "DISTRIBUTOR_PARTITIONER", os.environ.get("QUIPPER_DISTRIBUTOR_PARTITIONER", "kahypar")
    ).strip().lower()
    if mode not in {"auto", "fallback", "kahypar"}:
        mode = "kahypar"

    if mode == "fallback" or (mode == "auto" and platform.system() == "Darwin"):
        return {v: (v % k) for v in range(n_qubits)}

    import kahypar  # type: ignore[import]

    indices, nets, weights = hypergraph_to_kahypar(hyp, n_qubits)
    n_nets = len(indices) - 1

    if n_nets == 0 or not nets:
        return {v: 0 for v in range(n_qubits)}

    ctx = kahypar.Context()
    ctx.loadINIconfiguration(config_path)
    ctx.setK(k)
    ctx.setEpsilon(float(config.EPSILON))
    ctx.suppressOutput(True)

    hg = kahypar.Hypergraph(n_qubits, n_nets, indices, nets, k, [], weights)
    kahypar.partition(hg, ctx)

    return {v: hg.blockID(v) for v in range(n_qubits)}


def _ignore_last_seam(segments: list[Segment]) -> list[Segment]:
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
        wire_range=last.wire_range,
    )
    return result


def _upd_with(matching: Matching, seg: Segment) -> Segment:
    """Apply a partition matching (rename) to a segment."""
    new_part = {w: matching.get(b, b) for w, b in seg.partition.items()}
    return Segment(
        gates=seg.gates,
        hypergraph=seg.hypergraph,
        partition=new_part,
        seam=seg.seam,
        wire_range=seg.wire_range,
    )


def get_rho(n_wires: int, seg1: Segment, seg2: Segment) -> Fraction:
    """Compute seam cost between two adjacent segments."""
    hyp1, part1 = seg1.hypergraph, seg1.partition
    hyp2, part2 = seg2.hypergraph, seg2.partition

    changing = [w for w in range(n_wires) if w in part1 and w in part2 and part1[w] != part2[w]]

    def hedges(wire: Wire, hyp: Hypergraph) -> int:
        return len(hyp.get(wire, []))

    def total_hs(hyp: Hypergraph) -> int:
        return sum(len(v) for v in hyp.values())

    def weight(wire: Wire, hyp: Hypergraph) -> Fraction:
        total = total_hs(hyp)
        if total == 0:
            return Fraction(0)
        return Fraction(hedges(wire, hyp), total)

    total = Fraction(0)
    for w in changing:
        if hedges(w, hyp1) < hedges(w, hyp2):
            total += weight(w, hyp1)
        else:
            total += weight(w, hyp2)
    return total


def find_valley(segments: list[Segment]) -> tuple[list[Segment], list[Segment], list[Segment]]:
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


def match_partitions(part1: Partition, part2: Partition, k: int, n_wires: int) -> Matching:
    """Match blocks between adjacent partitions with branch-and-bound heuristic."""
    p1 = {w: b for w, b in part1.items() if w < n_wires}
    p2 = {w: b for w, b in part2.items() if w < n_wires}

    blocks1: dict[int, list[Wire]] = {b: [] for b in range(k)}
    for w, b in p1.items():
        blocks1[b].append(w)

    heuristic = _heuristic_cost(blocks1, p2)
    all_blocks = list(range(k))
    initial_cost = sum(heuristic.values())
    initial: list[tuple[Matching, int]] = [(dict(), initial_cost)]

    return _match_partitions_rec(blocks1, p2, heuristic, all_blocks, initial)


def _heuristic_cost(blocks1: dict[int, list[Wire]], part2: Partition) -> dict[int, int]:
    result: dict[int, int] = {}
    for block, wires in blocks1.items():
        if not wires:
            result[block] = 0
            continue
        valid = [w for w in wires if w in part2]
        if not valid:
            result[block] = 0
            continue
        from collections import Counter

        counts = Counter(part2[w] for w in valid)
        max_same = max(counts.values()) if counts else 0
        result[block] = len(wires) - max_same
    return result


def _match_partitions_rec(
    blocks1: dict[int, list[Wire]],
    part2: Partition,
    heuristic: dict[int, int],
    all_blocks: list[int],
    matchings: list[tuple[Matching, int]],
) -> Matching:
    while True:
        if not matchings:
            return {b: b for b in all_blocks}

        best_matching, best_cost = matchings[0]
        if len(best_matching) == len(all_blocks):
            return best_matching

        unmatched = [b for b in all_blocks if b not in best_matching]
        if not unmatched:
            return best_matching

        this_block = unmatched[0]
        this_wires = blocks1[this_block]

        already_used = set(best_matching.values())
        candidates = [b for b in all_blocks if b not in already_used]

        new_matchings: list[tuple[Matching, int]] = []
        for block in candidates:
            new_matching = dict(best_matching)
            new_matching[this_block] = block
            real_cost = sum(1 for w in this_wires if w in part2 and part2[w] != block)
            new_cost = best_cost - heuristic[this_block] + real_cost
            new_matchings.append((new_matching, new_cost))

        matchings = sorted(matchings[1:] + new_matchings, key=lambda x: x[1])


def compute_new_seams(n_wires: int, segments: list[Segment]) -> list[Segment]:
    """Assign rho values to Compute seams."""
    result: list[Segment] = []
    for i, seg in enumerate(segments):
        if isinstance(seg.seam, SeamCompute) and i + 1 < len(segments):
            new_seam: Seam = SeamValue(value=get_rho(n_wires, seg, segments[i + 1]))
        else:
            new_seam = seg.seam
        result.append(
            Segment(
                gates=seg.gates,
                hypergraph=seg.hypergraph,
                partition=seg.partition,
                seam=new_seam,
                wire_range=seg.wire_range,
            )
        )
    return result


def match_segments(
    k: int, n_wires: int, prev_matching: Matching, segments: list[Segment]
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
            new_matching = match_partitions(curr_seg.partition, prev_seg.partition, k, n_wires)
        else:
            new_matching = prev_matching

        updated = _upd_with(new_matching, curr_seg)
        result.append(updated)
        prev_matching = new_matching

    return result


def count_teles(part1: Partition, part2: Partition, n_wires: int) -> int:
    """Count wires that change block between two adjacent partitions."""
    return sum(1 for w in range(n_wires) if w in part1 and w in part2 and part1[w] != part2[w])


def merge_min(to_hyp: ToHyp, to_part: ToPart, n_wires: int, segments: list[Segment]) -> list[Segment]:
    """Find minima in rho sequence and try merging adjacent segments."""
    if not segments:
        return []

    head = segments[0]
    if isinstance(head.seam, SeamStop):
        return [head] + merge_min(to_hyp, to_part, n_wires, segments[1:])

    if not isinstance(head.seam, SeamValue):
        return segments

    before_min, after_min, rest = find_valley(segments)
    if len(after_min) < 2:
        marked = [
            Segment(
                gates=s.gates,
                hypergraph=s.hypergraph,
                partition=s.partition,
                seam=SeamStop(),
                wire_range=s.wire_range,
            )
            for s in after_min
        ]
        return before_min + marked + merge_min(to_hyp, to_part, n_wires, rest)

    left_seg, right_seg = after_min[0], after_min[1]
    merged_gates = left_seg.gates + right_seg.gates
    merged_hyp = to_hyp(merged_gates)
    merged_part = to_part(merged_hyp)
    merged_wire_range = (left_seg.wire_range[0], right_seg.wire_range[1])
    merged_seg = Segment(
        gates=merged_gates,
        hypergraph=merged_hyp,
        partition=merged_part,
        seam=SeamCompute(),
        wire_range=merged_wire_range,
    )

    cuts_left = count_cuts(left_seg)
    cuts_right = count_cuts(right_seg)
    teles = count_teles(left_seg.partition, right_seg.partition, n_wires)
    cuts_merged = count_cuts(merged_seg)

    if cuts_left + cuts_right + teles < cuts_merged:
        marked = [
            Segment(
                gates=s.gates,
                hypergraph=s.hypergraph,
                partition=s.partition,
                seam=SeamStop(),
                wire_range=s.wire_range,
            )
            for s in after_min
        ]
        return before_min + marked + merge_min(to_hyp, to_part, n_wires, rest)

    remaining_after = after_min[2:]
    return before_min + merge_min(to_hyp, to_part, n_wires, [merged_seg] + remaining_after + rest)


def merge_seams(to_hyp: ToHyp, to_part: ToPart, k: int, n_wires: int, segments: list[Segment]) -> list[Segment]:
    """Iteratively merge segments until all seams are Stop."""
    id_matching: Matching = {b: b for b in range(k)}

    while True:
        if all(isinstance(s.seam, SeamStop) for s in segments):
            return segments

        matched = match_segments(k, n_wires, id_matching, segments)
        seamed = compute_new_seams(n_wires, matched)
        merged = merge_min(to_hyp, to_part, n_wires, seamed)
        segments = _ignore_last_seam(merged)
