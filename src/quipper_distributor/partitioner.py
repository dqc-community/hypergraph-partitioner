"""KaHyPar bindings and segment management.

Port of Partitioner.hs.
"""

from __future__ import annotations

import os
import platform
from fractions import Fraction
from typing import Callable

from quipper_distributor.hgraph_builder import (
    build_hypergraph,
    count_cuts,
    hypergraph_to_kahypar,
)
from quipper_distributor.models.gate import Gate, Wire, is_cz
from quipper_distributor.models.hypergraph import Hypergraph, Matching, Partition
from quipper_distributor.models.segment import Seam, SeamCompute, SeamStop, SeamValue, Segment

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ToHyp = Callable[[list[Gate]], Hypergraph]
ToPart = Callable[[Hypergraph], Partition]


# ---------------------------------------------------------------------------
# KaHyPar integration
# ---------------------------------------------------------------------------


def partition_hypergraph(hyp: Hypergraph, n_qubits: int, k: int, config_path: str) -> Partition:
    """Partition hypergraph.

    Mode is controlled by QUIPPER_DISTRIBUTOR_PARTITIONER:
    - "auto" (default): use fallback on macOS, KaHyPar elsewhere
    - "fallback": always use deterministic round-robin assignment
    - "kahypar": always use KaHyPar
    """
    mode = os.environ.get("QUIPPER_DISTRIBUTOR_PARTITIONER", "auto").strip().lower()
    if mode not in {"auto", "fallback", "kahypar"}:
        mode = "auto"

    if mode == "fallback" or (mode == "auto" and platform.system() == "Darwin"):
        return {v: (v % k) for v in range(n_qubits)}

    import kahypar  # type: ignore[import]

    from quipper_distributor import config

    indices, nets, weights = hypergraph_to_kahypar(hyp, n_qubits)
    n_nets = len(indices) - 1

    if n_nets == 0 or not nets:
        # Trivial partition — assign all to block 0
        return {v: 0 for v in range(n_qubits)}

    ctx = kahypar.Context()
    ctx.loadINIconfiguration(config_path)
    ctx.setK(k)
    ctx.setEpsilon(float(config.EPSILON))
    ctx.suppressOutput(True)

    hg = kahypar.Hypergraph(n_qubits, n_nets, indices, nets, k, [], weights)
    kahypar.partition(hg, ctx)

    return {v: hg.blockID(v) for v in range(n_qubits)}


# ---------------------------------------------------------------------------
# Initial segments
# ---------------------------------------------------------------------------


def _seam_pos(n: int, gates: list[Gate]) -> int:
    """Find position in gates after the Nth CZ gate.

    Matches seamPos in Partitioner.hs.
    """
    if not gates:
        return 0
    # Position of first CZ gate (0-indexed)
    first_cz = 0
    while first_cz < len(gates) and not is_cz(gates[first_cz]):
        first_cz += 1
    if n == 0:
        # Position just after the first non-CZ prefix
        return first_cz
    if first_cz >= len(gates):
        return len(gates)
    return first_cz + 1 + _seam_pos(n - 1, gates[first_cz + 1 :])


def initial_segments(
    gates: list[Gate],
    init_seg_size: int,
    to_hyp: ToHyp,
    to_part: ToPart,
) -> list[Segment]:
    """Split gates into initial segments at CZ-gate boundaries.

    Each segment contains up to init_seg_size CZ gates.
    """
    segments: list[Segment] = []
    remaining = list(gates)
    seg_id = 0

    while remaining:
        split = _seam_pos(init_seg_size, remaining)
        if split == 0 and remaining:
            # No CZ gate found — take everything
            split = len(remaining)
        this_gates = remaining[:split]
        remaining = remaining[split:]

        hyp = to_hyp(this_gates)
        part = to_part(hyp)
        seg = Segment(
            gates=this_gates,
            hypergraph=hyp,
            partition=part,
            seam=SeamCompute(),
            wire_range=(seg_id, seg_id),
        )
        segments.append(seg)
        seg_id += 1

    return segments


# ---------------------------------------------------------------------------
# Seam management helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# getRho — matching getRho in Partitioner.hs
# ---------------------------------------------------------------------------


def get_rho(n_wires: int, seg1: Segment, seg2: Segment) -> Fraction:
    """Compute the seam cost between two adjacent segments.

    A low value means the partitions are similar (merge is cheap).
    """
    hyp1, part1 = seg1.hypergraph, seg1.partition
    hyp2, part2 = seg2.hypergraph, seg2.partition

    # Only consider wires that exist in both partitions and have changed block
    changing = [w for w in range(n_wires) if w in part1 and w in part2 and part1[w] != part2[w]]

    def hedges(wire: Wire, hyp: Hypergraph) -> int:
        return len(hyp.get(wire, []))

    def total_hs(hyp: Hypergraph) -> int:
        return sum(len(v) for v in hyp.values())

    def weight(wire: Wire, hyp: Hypergraph) -> Fraction:
        t = total_hs(hyp)
        if t == 0:
            return Fraction(0)
        return Fraction(hedges(wire, hyp), t)

    total = Fraction(0)
    for w in changing:
        h1 = hedges(w, hyp1)
        h2 = hedges(w, hyp2)
        if h1 < h2:
            total += weight(w, hyp1)
        else:
            total += weight(w, hyp2)
    return total


# ---------------------------------------------------------------------------
# findValley — matching findValleyRec in Partitioner.hs
# ---------------------------------------------------------------------------


def find_valley(segments: list[Segment]) -> tuple[list[Segment], list[Segment], list[Segment]]:
    """Find the minimum-rho valley in the seam sequence.

    Returns (before_min, valley_including_min, rest).
    """
    min_pos, end_pos = _find_valley_rec(segments, 0, None)
    before_min = segments[:min_pos]
    after_min = segments[min_pos : end_pos + 1]
    rest = segments[end_pos + 1 :]
    return before_min, after_min, rest


def _find_valley_rec(segments: list[Segment], pos: int, current_min: int | None) -> tuple[int, int]:
    """Recursive valley finder."""
    if len(segments) < 2:
        return (pos, pos)

    s = segments[0]
    next_s = segments[1]

    if not isinstance(s.seam, SeamValue):
        return (pos, pos)

    rho = s.seam.value
    next_seam = next_s.seam

    if current_min is None:
        # Haven't found a minimum yet
        if isinstance(next_seam, SeamValue):
            if rho < next_seam.value:
                # Current is a local minimum
                return _find_valley_rec(segments[1:], pos + 1, pos)
            else:
                return _find_valley_rec(segments[1:], pos + 1, None)
        elif isinstance(next_seam, SeamStop):
            return (pos, pos + 1)
        else:
            return (pos, pos)
    else:
        # We know the minimum, now find end of valley
        if isinstance(next_seam, SeamValue):
            if rho > next_seam.value:
                return (current_min, pos)
            else:
                return _find_valley_rec(segments[1:], pos + 1, current_min)
        elif isinstance(next_seam, SeamStop):
            return (current_min, pos + 1)
        else:
            return (current_min, pos)


# ---------------------------------------------------------------------------
# matchPartitions — heuristic block-matching
# ---------------------------------------------------------------------------


def match_partitions(part1: Partition, part2: Partition, k: int, n_wires: int) -> Matching:
    """Branch-and-bound heuristic to match blocks between partitions.

    Returns a mapping from blocks of part1 to blocks of part2
    so that the combined circuit needs fewest teleportations.
    """
    # Filter to wire vertices only
    p1 = {w: b for w, b in part1.items() if w < n_wires}
    p2 = {w: b for w, b in part2.items() if w < n_wires}

    # blocks1: block → list of wires in that block (from part1)
    blocks1: dict[int, list[Wire]] = {b: [] for b in range(k)}
    for w, b in p1.items():
        blocks1[b].append(w)

    heuristic = _heuristic_cost(blocks1, p2)
    all_blocks = list(range(k))
    initial_cost = sum(heuristic.values())
    initial: list[tuple[Matching, int]] = [(dict(), initial_cost)]

    return _match_partitions_rec(blocks1, p2, heuristic, all_blocks, initial)


def _heuristic_cost(blocks1: dict[int, list[Wire]], part2: Partition) -> dict[int, int]:
    """Minimum swaps needed to match each block from part1 to its best counterpart in part2."""
    result: dict[int, int] = {}
    for b, ws in blocks1.items():
        if not ws:
            result[b] = 0
            continue
        valid_ws = [w for w in ws if w in part2]
        if not valid_ws:
            result[b] = 0
            continue
        # Count frequency of each block assignment in part2 for wires in this block
        from collections import Counter

        counts = Counter(part2[w] for w in valid_ws)
        max_same = max(counts.values()) if counts else 0
        result[b] = len(ws) - max_same
    return result


def _match_partitions_rec(
    blocks1: dict[int, list[Wire]],
    part2: Partition,
    heuristic: dict[int, int],
    all_blocks: list[int],
    matchings: list[tuple[Matching, int]],
) -> Matching:
    """Branch-and-bound recursive matching search."""
    while True:
        if not matchings:
            # Fallback: identity matching
            return {b: b for b in all_blocks}

        # Best candidate is the one with lowest cost
        best_matching, best_cost = matchings[0]

        if len(best_matching) == len(all_blocks):
            return best_matching

        # Find a block from blocks1 not yet allocated
        unmatched = [b for b in all_blocks if b not in best_matching]
        if not unmatched:
            return best_matching

        this_block = unmatched[0]
        this_wires = blocks1[this_block]

        # Try all possible assignments for this_block
        already_used = set(best_matching.values())
        candidates = [b for b in all_blocks if b not in already_used]

        new_matchings: list[tuple[Matching, int]] = []
        for b in candidates:
            new_matching = dict(best_matching)
            new_matching[this_block] = b
            # Update cost: substitute heuristic estimate with real cost
            real_cost = sum(
                1 for w in this_wires if w in part2 and part2[w] is not None and part2[w] != b
            )
            new_cost = best_cost - heuristic[this_block] + real_cost
            new_matchings.append((new_matching, new_cost))

        # Merge new candidates with remaining and sort by cost
        matchings = sorted(matchings[1:] + new_matchings, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# compute_new_seams
# ---------------------------------------------------------------------------


def compute_new_seams(n_wires: int, segments: list[Segment]) -> list[Segment]:
    """Assign rho values to Compute seams."""
    result: list[Segment] = []
    for i, seg in enumerate(segments):
        if isinstance(seg.seam, SeamCompute) and i + 1 < len(segments):
            rho = get_rho(n_wires, seg, segments[i + 1])
            new_seam: Seam = SeamValue(value=rho)
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


# ---------------------------------------------------------------------------
# match_segments
# ---------------------------------------------------------------------------


def match_segments(
    k: int, n_wires: int, prev_matching: Matching, segments: list[Segment]
) -> list[Segment]:
    """Rename blocks so they match optimally between adjacent segments."""
    if not segments:
        return []

    result: list[Segment] = []
    seg0 = _upd_with(prev_matching, segments[0])
    result.append(seg0)

    for i in range(1, len(segments)):
        prev_seg = result[-1]
        curr_seg = segments[i]

        # Determine whether we need a fresh matching
        prev_seam = prev_seg.seam
        curr_seam = curr_seg.seam

        if isinstance(prev_seam, SeamCompute) or isinstance(curr_seam, SeamCompute):
            # Compute fresh matching from next (curr) to prev
            new_matching = match_partitions(curr_seg.partition, prev_seg.partition, k, n_wires)
        else:
            new_matching = prev_matching

        updated = _upd_with(new_matching, curr_seg)
        result.append(updated)
        prev_matching = new_matching

    return result


# ---------------------------------------------------------------------------
# count_teles — between two adjacent segments
# ---------------------------------------------------------------------------


def count_teles(part1: Partition, part2: Partition, n_wires: int) -> int:
    """Count wires that change block between two adjacent partitions."""
    return sum(1 for w in range(n_wires) if w in part1 and w in part2 and part1[w] != part2[w])


# ---------------------------------------------------------------------------
# merge_min
# ---------------------------------------------------------------------------


def merge_min(
    to_hyp: ToHyp,
    to_part: ToPart,
    n_wires: int,
    segments: list[Segment],
) -> list[Segment]:
    """Find minima in the rho sequence and try merging adjacent segments."""
    if not segments:
        return []

    head = segments[0]
    if isinstance(head.seam, SeamStop):
        return [head] + merge_min(to_hyp, to_part, n_wires, segments[1:])

    if not isinstance(head.seam, SeamValue):
        return segments  # Should not happen

    before_min, after_min, rest = find_valley(segments)

    if len(after_min) < 2:
        # Can't merge — mark as Stop
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

    left_seg = after_min[0]
    right_seg = after_min[1]

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

    # Check if merging is beneficial
    cuts_left = count_cuts(left_seg)
    cuts_right = count_cuts(right_seg)
    teles = count_teles(left_seg.partition, right_seg.partition, n_wires)
    cuts_merged = count_cuts(merged_seg)

    if cuts_left + cuts_right + teles < cuts_merged:
        # Keep original segments, mark seams as Stop
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
    else:
        # Replace the two merged segments with the merged one
        remaining_after = after_min[2:]
        return before_min + merge_min(
            to_hyp, to_part, n_wires, [merged_seg] + remaining_after + rest
        )


# ---------------------------------------------------------------------------
# merge_seams — iterative until all seams are Stop
# ---------------------------------------------------------------------------


def merge_seams(
    to_hyp: ToHyp,
    to_part: ToPart,
    k: int,
    n_wires: int,
    segments: list[Segment],
) -> list[Segment]:
    """Iteratively merge segments until all seams are Stop."""
    id_matching: Matching = {b: b for b in range(k)}

    while True:
        if all(isinstance(s.seam, SeamStop) for s in segments):
            return segments

        matched = match_segments(k, n_wires, id_matching, segments)
        seamed = compute_new_seams(n_wires, matched)
        merged = merge_min(to_hyp, to_part, n_wires, seamed)
        segments = _ignore_last_seam(merged)


# ---------------------------------------------------------------------------
# Top-level partitioner
# ---------------------------------------------------------------------------


def partitioner(
    k: int,
    init_seg_size: int,
    max_hedge_dist: int,
    config_path: str,
    n_qubits: int,
    n_wires: int,
    gates: list[Gate],
) -> list[Segment]:
    """Full partitioning pipeline.

    Returns a list of Segment objects with final partitions.
    """
    to_hyp: ToHyp = lambda gs: build_hypergraph(gs, n_qubits, max_hedge_dist)
    to_part: ToPart = lambda hyp: partition_hypergraph(hyp, n_qubits, k, config_path)

    init_segs = initial_segments(gates, init_seg_size, to_hyp, to_part)
    init_segs = _ignore_last_seam(init_segs)

    return merge_seams(to_hyp, to_part, k, n_wires, init_segs)
