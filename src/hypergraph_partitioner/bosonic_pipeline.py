"""Bosonic-model-native stats pipeline for partitioning metrics.

This module intentionally keeps the proven seam/merge heuristics while switching
runtime data flow to bosonic_model instructions.
"""

from __future__ import annotations

from collections.abc import Iterable

from bosonic_model import BarrierInstruction, Circuit, ConditionalInstruction
from bosonic_model.instructions import CzInstruction, InstructionType

from hypergraph_partitioner.cz_commutation import push_cz_early
from hypergraph_partitioner.models.annotated import (
    AnnotatedOp,
    BlockId,
    BoundaryId,
    BoundaryTeleportOp,
    LocalOp,
    NonlocalCZOp,
    PartitionedCircuit,
    PartitionedSegment,
    SegmentBoundary,
    SegmentId,
    TeleportBoundary,
    WireId,
)
from hypergraph_partitioner.models.hypergraph import Hypergraph, InteractionVertex, WireVertex
from hypergraph_partitioner.models.segment import SeamCompute, Segment
from hypergraph_partitioner.partitioner import _ignore_last_seam, merge_seams, partition_hypergraph
from hypergraph_partitioner.qiskit_normalization import normalize_to_one_qubit_and_cz


def _unwrap_conditional(inst: InstructionType) -> InstructionType:
    if isinstance(inst, ConditionalInstruction):
        return inst.op
    return inst


def _is_interaction(inst: InstructionType) -> bool:
    inner = _unwrap_conditional(inst)
    if isinstance(inner, BarrierInstruction):
        return False
    qubits = list(getattr(inner, "qubits", []) or [])
    return len(qubits) >= 2


def _interaction_wires(inst: InstructionType) -> list[int]:
    return list(getattr(_unwrap_conditional(inst), "qubits", []) or [])


def _target_wire(inst: InstructionType) -> int | None:
    inner = _unwrap_conditional(inst)
    qubits = list(getattr(inner, "qubits", []) or [])
    if len(qubits) == 1:
        return qubits[0]
    qubit = getattr(inner, "qubit", None)
    return qubit if isinstance(qubit, int) else None


def prepare_instructions(instructions: Iterable[InstructionType]) -> list[InstructionType]:
    """Phase-2 minimal preparation: drop barriers, preserve instruction order."""
    return [inst for inst in instructions if not isinstance(_unwrap_conditional(inst), BarrierInstruction)]


def build_hypergraph_from_instructions(
    instructions: list[InstructionType], n_qubits: int, max_hedge_dist: int
) -> Hypergraph:
    """Build hypergraph directly from bosonic instructions."""
    del max_hedge_dist

    wires = {wire_id: WireVertex(wire_id=wire_id) for wire_id in range(n_qubits)}
    interactions: dict[int, InteractionVertex] = {}
    interaction_id = 0

    for position, inst in enumerate(instructions):
        if not _is_interaction(inst):
            continue
        qubits = tuple(_interaction_wires(inst))
        interactions[interaction_id] = InteractionVertex(
            interaction_id=interaction_id,
            position=position,
            qubits=qubits,
        )
        interaction_id += 1

    return Hypergraph(wires=wires, interactions=interactions)


def _interaction_seam_pos(n: int, instructions: list[InstructionType]) -> int:
    if not instructions:
        return 0

    first = 0
    while first < len(instructions) and not _is_interaction(instructions[first]):
        first += 1

    if n == 0:
        return first
    if first >= len(instructions):
        return len(instructions)

    return first + 1 + _interaction_seam_pos(n - 1, instructions[first + 1 :])


def _initial_segments(
    instructions: list[InstructionType],
    init_seg_size: int,
    n_qubits: int,
    k: int,
    max_hedge_dist: int,
    config_path: str,
) -> list[Segment]:
    segments: list[Segment] = []
    remaining = list(instructions)
    seg_id = 0

    while remaining:
        split = _interaction_seam_pos(init_seg_size, remaining)
        if split == 0 and remaining:
            split = len(remaining)

        this_insts = remaining[:split]
        remaining = remaining[split:]

        hyp = build_hypergraph_from_instructions(this_insts, n_qubits, max_hedge_dist)
        part = partition_hypergraph(hyp, n_qubits, k, config_path)

        segments.append(
            Segment(
                gates=this_insts,
                hypergraph=hyp,
                partition=part,
                seam=SeamCompute(),
                wire_range=(seg_id, seg_id),
            )
        )
        seg_id += 1

    return segments


def partition_circuit(
    circuit: Circuit,
    *,
    k: int,
    init_seg_size: int,
    max_hedge_dist: int,
    config_path: str,
) -> PartitionedCircuit:
    normalized = _preprocess(circuit)
    
    instructions = prepare_instructions(normalized.instructions)
    n_qubits = circuit.qubits()

    initial = _initial_segments(
        instructions,
        init_seg_size,
        n_qubits,
        k,
        max_hedge_dist,
        config_path,
    )
    initial = _ignore_last_seam(initial)

    def to_hyp(insts: list[InstructionType]) -> Hypergraph:
        return build_hypergraph_from_instructions(insts, n_qubits, max_hedge_dist)

    def to_part(hyp: Hypergraph) -> dict[int, int]:
        return partition_hypergraph(hyp, n_qubits, k, config_path)

    merged = merge_seams(to_hyp, to_part, k, n_qubits, max_hedge_dist, initial)
    return _annotate_partitioned_circuit(merged)


def _preprocess(circuit: Circuit) -> Circuit:
    res = normalize_to_one_qubit_and_cz(circuit)
    instructions = push_cz_early(res.instructions)
    res = Circuit(qregs=res.qregs, cregs=res.cregs, instructions=instructions)
    return res


def count_nonlocal_interactions(circuit: PartitionedCircuit) -> int:
    return sum(isinstance(op, NonlocalCZOp) for op in circuit.operations)


def count_interactions(instructions: Iterable[InstructionType]) -> int:
    return sum(1 for inst in instructions if _is_interaction(inst))


def count_teleports(circuit: PartitionedCircuit) -> int:
    return sum(len(boundary.teleports) for boundary in circuit.boundaries)


def _annotate_partitioned_circuit(segments: list[Segment]) -> PartitionedCircuit:
    public_segments = [_to_partitioned_segment(seg, idx) for idx, seg in enumerate(segments)]
    boundaries: list[SegmentBoundary] = []
    operations: list[AnnotatedOp] = []

    for idx, seg in enumerate(public_segments):
        operations.extend(_annotate_segment_ops(seg))
        if idx + 1 < len(public_segments):
            boundary = _build_boundary(seg, public_segments[idx + 1], idx)
            boundaries.append(boundary)
            operations.extend(
                BoundaryTeleportOp(
                    boundary_id=boundary.boundary_id,
                    wire=teleport.wire,
                    from_block=teleport.from_block,
                    to_block=teleport.to_block,
                )
                for teleport in boundary.teleports
            )

    return PartitionedCircuit(
        segments=public_segments,
        boundaries=boundaries,
        operations=operations,
    )


def _to_partitioned_segment(seg: Segment, idx: int) -> PartitionedSegment:
    return PartitionedSegment(
        segment_id=SegmentId(idx),
        instructions=seg.gates,
        partition={WireId(w): BlockId(b) for w, b in seg.partition.items()},
    )


def _build_boundary(
    left: PartitionedSegment, right: PartitionedSegment, boundary_idx: int
) -> SegmentBoundary:
    teleports = [
        TeleportBoundary(
            wire=wire,
            from_block=left.partition[wire],
            to_block=right.partition[wire],
        )
        for wire in left.partition
        if wire in right.partition and left.partition[wire] != right.partition[wire]
    ]
    return SegmentBoundary(
        boundary_id=BoundaryId(boundary_idx),
        left_segment_id=left.segment_id,
        right_segment_id=right.segment_id,
        teleports=teleports,
    )


def _annotate_segment_ops(seg: PartitionedSegment) -> list[AnnotatedOp]:
    result: list[AnnotatedOp] = []
    for inst in seg.instructions:
        inner = _unwrap_conditional(inst)
        qubits = tuple(_interaction_wires(inst) if _is_interaction(inst) else (getattr(inner, "qubits", []) or []))
        blocks = tuple(seg.partition[WireId(w)] for w in qubits if WireId(w) in seg.partition)

        if isinstance(inner, CzInstruction):
            control_wire = WireId(inner.control)
            target_wire = WireId(inner.target)
            control_block = seg.partition[control_wire]
            target_block = seg.partition[target_wire]
            if control_block != target_block:
                result.append(
                    NonlocalCZOp(
                        segment_id=seg.segment_id,
                        instruction=inner,
                        control_wire=control_wire,
                        target_wire=target_wire,
                        control_block=control_block,
                        target_block=target_block,
                    )
                )
                continue

        result.append(LocalOp(segment_id=seg.segment_id, instruction=inst, blocks=blocks))

    return result
