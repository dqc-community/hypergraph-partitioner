"""Public API for bosonic-model partitioning and annotation."""

from hypergraph_partitioner.bosonic_pipeline import (
    partition_circuit,
)
from hypergraph_partitioner.distributor import build_annotated_circuit
from hypergraph_partitioner.circuit_lowering import (
    lower_distributed_circuit,
)
from hypergraph_partitioner.models.circuit_annotations import (
    AnnotatedOp,
    BoundarySwapOp,
    BoundaryTeleportOp,
    LocalOp,
    NonlocalCZOp,
    PartitionedCircuit,
    PartitionedSegment,
    SegmentBoundary,
    SwapBoundary,
    TeleportBoundary,
)

__all__ = [
    "partition_circuit",
    "build_annotated_circuit",
    "lower_distributed_circuit",
    "PartitionedCircuit",
    "PartitionedSegment",
    "SegmentBoundary",
    "TeleportBoundary",
    "SwapBoundary",
    "AnnotatedOp",
    "LocalOp",
    "NonlocalCZOp",
    "BoundaryTeleportOp",
    "BoundarySwapOp",
]
