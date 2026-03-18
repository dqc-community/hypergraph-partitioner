"""KaHyPar-backed hypergraph partitioning helpers."""

from __future__ import annotations

from hypergraph_partitioner import config
from hypergraph_partitioner.hgraph_builder import hypergraph_to_kahypar
from hypergraph_partitioner.models.hypergraph import Hypergraph, Partition

import kahypar


def partition_hypergraph(hyp: Hypergraph, n_qubits: int, k: int, config_path: str) -> Partition:
    """Partition hypergraph."""

    indices, nets, weights = hypergraph_to_kahypar(hyp, n_qubits)
    n_nets = len(indices) - 1

    if n_nets == 0 or not nets:
        return _balanced_fallback_partition(n_qubits, k)

    ctx = kahypar.Context()
    ctx.loadINIconfiguration(config_path)
    ctx.setK(k)
    ctx.setEpsilon(float(config.EPSILON))
    ctx.suppressOutput(True)

    hg = kahypar.Hypergraph(n_qubits, n_nets, indices, nets, k, [], weights)
    kahypar.partition(hg, ctx)

    return {v: hg.blockID(v) for v in range(n_qubits)}

def _balanced_fallback_partition(n_qubits: int, k: int) -> Partition:
    if k < 1:
        raise ValueError("k must be positive")
    return {qubit: qubit % k for qubit in range(n_qubits)}
