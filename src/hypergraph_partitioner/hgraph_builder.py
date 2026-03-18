"""Hypergraph utility helpers used by partitioning."""

from __future__ import annotations

from hypergraph_partitioner.models.hypergraph import Hypergraph


def hypergraph_to_kahypar(hyp: Hypergraph, n_qubits: int) -> tuple[list[int], list[int], list[int]]:
    """Convert hypergraph to CSR format for the kahypar Python API."""
    flat_data: list[list[int]] = []
    for interaction in sorted(hyp.interactions.values(), key=lambda interaction: interaction.position):
        vertices = sorted(v for v in interaction.qubits if 0 <= v < n_qubits)
        if len(vertices) >= 2:
            flat_data.append(vertices)

    if not flat_data:
        return [0], [], [1] * n_qubits

    hyperedge_indices: list[int] = [0]
    hyperedge_vertices: list[int] = []
    for hedge_verts in flat_data:
        hyperedge_vertices.extend(hedge_verts)
        hyperedge_indices.append(len(hyperedge_vertices))

    vertex_weights = [1] * n_qubits
    return hyperedge_indices, hyperedge_vertices, vertex_weights
