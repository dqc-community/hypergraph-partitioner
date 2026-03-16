"""Public API for bosonic-model partitioning and annotation."""

from hypergraph_partitioner.bosonic_pipeline import (
    count_interactions,
    count_nonlocal_interactions,
    count_teleports,
    partition_circuit,
)
from hypergraph_partitioner.lowering import (
    annotated_to_distributed_circuit,
    lower_distributed_circuit,
)
from hypergraph_partitioner.models.annotated import (
    AnnotatedOp,
    BoundaryTeleportOp,
    LocalOp,
    NonlocalCZOp,
    PartitionedCircuit,
    PartitionedSegment,
    SegmentBoundary,
    TeleportBoundary,
)

__all__ = [
    "partition_circuit",
    "annotated_to_distributed_circuit",
    "lower_distributed_circuit",
    "count_interactions",
    "count_nonlocal_interactions",
    "count_teleports",
    "PartitionedCircuit",
    "PartitionedSegment",
    "SegmentBoundary",
    "TeleportBoundary",
    "AnnotatedOp",
    "LocalOp",
    "NonlocalCZOp",
    "BoundaryTeleportOp",
]
