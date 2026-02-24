"""Hypergraph utility helpers used by partitioning."""

from __future__ import annotations

from quipper_distributor.models.hypergraph import Hedge, Hypergraph
from quipper_distributor.models.segment import Segment


def _split_long_hedges(hedges: list[Hedge], max_dist: int) -> list[Hedge]:
    """Recursively split hedges whose span exceeds ``max_dist``."""
    if not hedges:
        return []

    result: list[Hedge] = []
    queue = list(hedges)

    while queue:
        hedge = queue.pop(0)
        if max_dist == 1:
            for w, p in hedge.wires:
                result.append(Hedge(nan=p, wires=[(w, p)], out_pos=p + 1))
        elif max_dist < (hedge.out_pos - hedge.nan) and len(hedge.wires) > 1:
            split = hedge.nan + (hedge.out_pos - hedge.nan) // 2
            before = [(w, p) for w, p in hedge.wires if p < split]
            after = [(w, p) for w, p in hedge.wires if p >= split]
            left = Hedge(nan=hedge.nan, wires=before, out_pos=split)
            right = Hedge(nan=split, wires=after, out_pos=hedge.out_pos)
            queue.insert(0, right)
            queue.insert(0, left)
        else:
            result.append(hedge)

    return result


def count_cuts(segment: Segment) -> int:
    """Count the number of hyperedge cuts in a segment."""
    total = 0
    for wire, hedges in segment.hypergraph.items():
        for hedge in hedges:
            vertex_set = [wire] + [w for w, _ in hedge.wires]
            blocks = {segment.partition[w] for w in vertex_set if w in segment.partition}
            total += max(0, len(blocks) - 1)
    return total


def hypergraph_to_kahypar(hyp: Hypergraph, n_qubits: int) -> tuple[list[int], list[int], list[int]]:
    """Convert hypergraph to CSR format for the kahypar Python API."""
    flat_data: list[list[int]] = []
    for wire, hedges in hyp.items():
        for hedge in hedges:
            vertices = [wire] + [w for w, _ in hedge.wires]
            seen: set[int] = set()
            unique: list[int] = []
            for vtx in vertices:
                if vtx not in seen:
                    seen.add(vtx)
                    unique.append(vtx)
            flat_data.append(unique)

    if not flat_data:
        return [0, 0], [], [1] * n_qubits

    hyperedge_indices: list[int] = [0]
    hyperedge_vertices: list[int] = []
    for hedge_verts in flat_data:
        valid = [v for v in hedge_verts if 0 <= v < n_qubits]
        hyperedge_vertices.extend(valid)
        hyperedge_indices.append(len(hyperedge_vertices))

    vertex_weights = [1] * n_qubits
    return hyperedge_indices, hyperedge_vertices, vertex_weights
