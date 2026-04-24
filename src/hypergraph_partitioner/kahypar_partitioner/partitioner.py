"""KaHyPar-backed hypergraph partitioning helpers."""

from __future__ import annotations

from hypergraph_partitioner import config
from hypergraph_partitioner.models.hypergraph import Hypergraph, Partition

try:
    import kahypar
except ImportError:
    raise ImportError(
        "kahypar is not available on Windows. Use WSL or switch to a different distributor."
    )


def partition_hypergraph(hyp: Hypergraph, n_qubits: int, nodes: int, config_path: str) -> Partition:
    """Partition hypergraph."""

    indices, nets, weights = _hypergraph_to_kahypar(hyp, n_qubits)
    n_nets = len(indices) - 1

    if n_nets == 0 or not nets:
        return _balanced_fallback_partition(n_qubits, nodes)

    ctx = kahypar.Context()
    ctx.loadINIconfiguration(config_path)
    ctx.setK(nodes)
    ctx.setEpsilon(float(config.EPSILON))
    ctx.suppressOutput(True)

    hg = kahypar.Hypergraph(n_qubits, n_nets, indices, nets, nodes, [], weights)
    kahypar.partition(hg, ctx)

    return {v: hg.blockID(v) for v in range(n_qubits)}


def _hypergraph_to_kahypar(hyp: Hypergraph, n_qubits: int) -> tuple[list[int], list[int], list[int]]:
    """Convert a hypergraph to CSR-style inputs for the KaHyPar Python API."""
    flat_data: list[list[int]] = []
    for interaction in sorted(hyp.interactions.values(), key=lambda interaction: interaction.position):
        vertices = sorted(qubit for qubit in interaction.qubits if 0 <= qubit < n_qubits)
        if len(vertices) >= 2:
            flat_data.append(vertices)

    if not flat_data:
        return [0], [], [1] * n_qubits

    hyperedge_indices: list[int] = [0]
    hyperedge_vertices: list[int] = []
    for hedge_vertices in flat_data:
        hyperedge_vertices.extend(hedge_vertices)
        hyperedge_indices.append(len(hyperedge_vertices))

    vertex_weights = [1] * n_qubits
    return hyperedge_indices, hyperedge_vertices, vertex_weights


def _balanced_fallback_partition(n_qubits: int, nodes: int) -> Partition:
    if nodes < 1:
        raise ValueError("nodes must be positive")
    return {qubit: qubit % nodes for qubit in range(n_qubits)}
