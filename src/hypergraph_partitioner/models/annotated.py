"""Annotated partitioned-circuit IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, TypeAlias

from bosonic_model.instructions import CzInstruction, InstructionType

WireId = NewType("WireId", int)
BlockId = NewType("BlockId", int)
SegmentId = NewType("SegmentId", int)
BoundaryId = NewType("BoundaryId", int)


@dataclass(frozen=True)
class PartitionedSegment:
    segment_id: SegmentId
    instructions: list[InstructionType]
    partition: dict[WireId, BlockId]


@dataclass(frozen=True)
class TeleportBoundary:
    wire: WireId
    from_block: BlockId
    to_block: BlockId


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
    blocks: tuple[BlockId, ...]


@dataclass(frozen=True)
class NonlocalCZOp:
    segment_id: SegmentId
    instruction: CzInstruction
    control_wire: WireId
    target_wire: WireId
    control_block: BlockId
    target_block: BlockId


@dataclass(frozen=True)
class BoundaryTeleportOp:
    boundary_id: BoundaryId
    wire: WireId
    from_block: BlockId
    to_block: BlockId


AnnotatedOp: TypeAlias = LocalOp | NonlocalCZOp | BoundaryTeleportOp


@dataclass(frozen=True)
class PartitionedCircuit:
    segments: list[PartitionedSegment]
    boundaries: list[SegmentBoundary]
    operations: list[AnnotatedOp]
