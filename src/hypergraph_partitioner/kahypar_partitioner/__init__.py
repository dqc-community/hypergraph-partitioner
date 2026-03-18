"""KaHyPar-backed partitioning helpers."""

from hypergraph_partitioner.kahypar_partitioner.partitioner import (
    _balanced_fallback_partition,
    partition_hypergraph,
)

__all__ = [
    "partition_hypergraph",
    "_balanced_fallback_partition",
]
