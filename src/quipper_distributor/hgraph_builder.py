"""Hypergraph construction from CZ gates.

Direct port of HGraphBuilder.hs.
"""

from __future__ import annotations

from quipper_distributor.models.gate import Gate, Wire, get_wires, is_classical, is_cz, target_of
from quipper_distributor.models.hypergraph import Hedge, Hypergraph, Partition
from quipper_distributor.models.segment import Segment


def build_hypergraph(gates: list[Gate], n_qubits: int, max_hedge_dist: int) -> Hypergraph:
    """Build a hypergraph from a gate list.

    Steps (matching buildHyp in HGraphBuilder.hs):
    1. _build_hyp_rec  — raw hypergraph from gate sequence
    2. _to_positive    — remap negative ancilla wire indices
    3. _split_long_hedges — split hedges exceeding max_hedge_dist
    4. Remove singleton (empty) hyperedges
    """
    hyp = _build_hyp_rec(gates, 0, 0)
    hyp = _to_positive(hyp, n_qubits)
    hyp = {w: _split_long_hedges(hs, max_hedge_dist) for w, hs in hyp.items()}
    hyp = {w: [h for h in hs if h.wires] for w, hs in hyp.items()}
    return hyp


def _to_positive(hyp: Hypergraph, n_qubits: int) -> Hypergraph:
    """Convert negative auxiliary wire indices to positive ones."""
    result: Hypergraph = {}
    for v, hedges in hyp.items():
        new_hedges = []
        for h in hedges:
            new_wires = [(n_qubits - w - 1, p) for w, p in h.wires]
            new_hedges.append(Hedge(nan=h.nan, wires=new_wires, out_pos=h.out_pos))
        result[v] = new_hedges
    return result


def _split_long_hedges(hedges: list[Hedge], max_dist: int) -> list[Hedge]:
    """Recursively split hedges whose span exceeds max_dist."""
    if not hedges:
        return []

    result: list[Hedge] = []
    queue = list(hedges)

    while queue:
        h = queue.pop(0)
        if max_dist == 1:
            # Standard graph partitioning: each (wire, pos) becomes its own hedge
            for w, p in h.wires:
                result.append(Hedge(nan=p, wires=[(w, p)], out_pos=p + 1))
        elif max_dist < (h.out_pos - h.nan) and len(h.wires) > 1:
            # Split at midpoint
            split = h.nan + (h.out_pos - h.nan) // 2
            before = [(w, p) for w, p in h.wires if p < split]
            after = [(w, p) for w, p in h.wires if p >= split]
            left = Hedge(nan=h.nan, wires=before, out_pos=split)
            right = Hedge(nan=split, wires=after, out_pos=h.out_pos)
            queue.insert(0, right)
            queue.insert(0, left)
        else:
            result.append(h)

    return result


def _build_hyp_rec(gates: list[Gate], pos: int, cz_vertex: int) -> Hypergraph:
    """Build hypergraph from gate list (iterative, matching buildHypRec).

    The hypergraph is built from the END of the circuit to the START,
    so we process gates in reverse order.
    """
    hyp: dict[Wire, list[Hedge]] = {}

    # Process in reverse order (from end to start, as in Haskell)
    for gate in reversed(gates):
        if is_classical(gate):
            pos += 1
            continue

        if is_cz(gate):
            wires_of_cz = get_wires(gate)  # [target] + controls
            # For each wire in the CZ, extend/create a hedge
            for w in wires_of_cz:
                if w not in hyp:
                    hyp[w] = [Hedge(nan=0, wires=[(cz_vertex - 1, pos)], out_pos=pos + 1)]
                else:
                    # Add czVertex to the last hedge group on that wire
                    last = hyp[w][-1]
                    new_wires = [(cz_vertex - 1, pos)] + last.wires
                    hyp[w][-1] = Hedge(nan=last.nan, wires=new_wires, out_pos=last.out_pos)
            cz_vertex -= 1
            pos += 1

        else:
            t = target_of(gate)
            if t is not None:
                if t not in hyp:
                    hyp[t] = [Hedge(nan=0, wires=[], out_pos=pos)]
                else:
                    # Close current hedge and start a new empty one
                    last = hyp[t][-1]
                    new_hedge = Hedge(nan=0, wires=[], out_pos=pos)
                    # The old last becomes (nan=[], pos=pos, old_wires, old_out_pos)
                    updated_last = Hedge(nan=0, wires=last.wires, out_pos=last.out_pos)
                    hyp[t] = hyp[t][:-1] + [updated_last, new_hedge]
            pos += 1

    return hyp


# ---------------------------------------------------------------------------
# countCuts — matching countCuts in Common.hs
# ---------------------------------------------------------------------------


def count_cuts(segment: Segment) -> int:
    """Count the number of hyperedge cuts in a segment."""
    hyp = segment.hypergraph
    part = segment.partition
    total = 0
    for v, hedges in hyp.items():
        for h in hedges:
            # vertices in this hedge: key wire v + all wires in h.wires
            vertex_set = [v] + [w for w, _ in h.wires]
            blocks = set()
            for w in vertex_set:
                if w in part:
                    blocks.add(part[w])
            total += max(0, len(blocks) - 1)
    return total


# ---------------------------------------------------------------------------
# hypergraph_to_kahypar — CSR format for kahypar Python package
# ---------------------------------------------------------------------------


def hypergraph_to_kahypar(
    hyp: Hypergraph, n_qubits: int
) -> tuple[list[int], list[int], list[int]]:
    """Convert hypergraph to CSR format for the kahypar Python API.

    Returns (hyperedge_indices, hyperedge_vertices, vertex_weights).

    Matches flatData logic from hypToString in HGraphBuilder.hs (Kahypar variant).
    """
    # Build flat data: for each hedge, collect unique vertices (1-indexed in Haskell, 0-indexed here)
    flat_data: list[list[int]] = []
    for v, hedges in hyp.items():
        for h in hedges:
            vertices = [v] + [w for w, _ in h.wires]
            # Remove duplicates while preserving order
            seen: set[int] = set()
            unique: list[int] = []
            for vtx in vertices:
                if vtx not in seen:
                    seen.add(vtx)
                    unique.append(vtx)
            flat_data.append(unique)

    if not flat_data:
        # Empty hypergraph: single trivial hedge
        return [0, 0], [], [1] * n_qubits

    # Determine number of vertices
    n_vertices = n_qubits  # only qubit vertices are partitioned

    # Build CSR
    hyperedge_indices: list[int] = [0]
    hyperedge_vertices: list[int] = []
    for hedge_verts in flat_data:
        # Filter to only include vertices in range [0, n_vertices)
        valid = [v for v in hedge_verts if 0 <= v < n_vertices]
        hyperedge_vertices.extend(valid)
        hyperedge_indices.append(len(hyperedge_vertices))

    # Vertex weights: 1 for qubit wires, 0 for ancilla
    vertex_weights = [1] * n_qubits

    return hyperedge_indices, hyperedge_vertices, vertex_weights
