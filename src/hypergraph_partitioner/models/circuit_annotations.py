"""Annotated partitioned-circuit IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, TypeAlias

from bosonic_model.instructions import CzInstruction, InstructionType

QubitId = NewType("QubitId", int)
NodeId = NewType("NodeId", int)
SegmentId = NewType("SegmentId", int)
BoundaryId = NewType("BoundaryId", int)


@dataclass(frozen=True)
class PartitionedSegment:
    segment_id: SegmentId
    instructions: list[InstructionType]
    partition: dict[QubitId, NodeId]


@dataclass(frozen=True)
class TeleportBoundary:
    qubit: QubitId
    from_node: NodeId
    to_node: NodeId


@dataclass(frozen=True)
class SegmentBoundary:
    boundary_id: BoundaryId
    left_segment_id: SegmentId
    right_segment_id: SegmentId
    teleports: list[TeleportBoundary]


@dataclass(frozen=True)
class LocalOp:
    segment_id: SegmentId
    instruction: InstructionType
    nodes: tuple[NodeId, ...]


@dataclass(frozen=True)
class NonlocalCZOp:
    segment_id: SegmentId
    instruction: CzInstruction
    control_qubit: QubitId
    target_qubit: QubitId
    control_node: NodeId
    target_node: NodeId


@dataclass(frozen=True)
class BoundaryTeleportOp:
    boundary_id: BoundaryId
    qubit: QubitId
    from_node: NodeId
    to_node: NodeId


AnnotatedOp: TypeAlias = LocalOp | NonlocalCZOp | BoundaryTeleportOp


@dataclass(frozen=True)
class PartitionedCircuit:
    segments: list[PartitionedSegment]
    boundaries: list[SegmentBoundary]
