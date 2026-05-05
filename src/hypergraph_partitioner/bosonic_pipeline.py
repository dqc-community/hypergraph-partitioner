"""Bosonic-model-native stats pipeline for partitioning metrics.

This module intentionally keeps the proven seam/merge heuristics while switching
runtime data flow to bosonic_model instructions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Literal

from bosonic_model import BarrierInstruction, Circuit, ConditionalInstruction, DistributedCircuit
from bosonic_model.instructions import CzInstruction, InstructionType

from hypergraph_partitioner.preprocessing.cz_commutation import push_cz_early
from hypergraph_partitioner.config import DEFAULT_CONFIG_PATH
from hypergraph_partitioner.models.circuit_annotations import (
    AnnotatedOp,
    NodeId,
    BoundaryId,
    BoundarySwapOp,
    BoundaryTeleportOp,
    LocalOp,
    NonlocalCZOp,
    PartitionedCircuit,
    PartitionedSegment,
    SegmentBoundary,
    SegmentId,
    SwapBoundary,
    QubitId,
)
from hypergraph_partitioner.models.hypergraph import Hypergraph, InteractionVertex, QubitVertex
from hypergraph_partitioner.models.segment import SeamCompute, Segment
from hypergraph_partitioner.segment_merger import ignore_last_seam, merge_seams, project_boundary_swaps
from hypergraph_partitioner.kahypar_partitioner import partition_hypergraph
from hypergraph_partitioner.preprocessing.normalization import normalize_to_one_qubit_and_cz


def partition_circuit(
    circuit: Circuit,
    *,
    nodes: int,
    qubits_per_node: int,
    init_seg_size: int = 10,
    max_hedge_dist: int = 100,
    output: Literal["symbolic", "lowered"] = "symbolic",
) -> DistributedCircuit:
    from hypergraph_partitioner.distributor import build_annotated_circuit

    partitioned = _partition_to_partitioned_circuit(
        circuit,
        nodes=nodes,
        init_seg_size=init_seg_size,
        max_hedge_dist=max_hedge_dist,
    )
    symbolic = build_annotated_circuit(partitioned, qubits_per_node=qubits_per_node)

    if output == "symbolic":
        return symbolic
    if output == "lowered":
        from hypergraph_partitioner.circuit_lowering import lower_distributed_circuit
        return lower_distributed_circuit(symbolic)
    else:
        raise ValueError(f"unsupported output mode: {output}")


def _partition_to_partitioned_circuit(
    circuit: Circuit,
    *,
    nodes: int,
    init_seg_size: int = 10,
    max_hedge_dist: int = 100,
) -> PartitionedCircuit:
    normalized = _preprocess(circuit)

    instructions = _prepare_instructions(normalized.instructions)
    n_qubits = circuit.qubits()

    initial = _initial_segments(
        instructions,
        init_seg_size,
        n_qubits,
        nodes,
        max_hedge_dist,
    )
    initial = ignore_last_seam(initial)

    def to_hyp(insts: list[InstructionType]) -> Hypergraph:
        return _build_hypergraph_from_instructions(insts, n_qubits)

    def to_part(hyp: Hypergraph) -> dict[int, int]:
        return partition_hypergraph(hyp, n_qubits, nodes, DEFAULT_CONFIG_PATH)

    merged = merge_seams(to_hyp, to_part, nodes, n_qubits, max_hedge_dist, initial)
    return _annotate_partitioned_circuit(merged)


def _preprocess(circuit: Circuit) -> Circuit:
    res = normalize_to_one_qubit_and_cz(circuit)
    instructions = push_cz_early(res.instructions)
    res = Circuit(qregs=res.qregs, cregs=res.cregs, instructions=instructions)
    return res


def _prepare_instructions(instructions: Iterable[InstructionType]) -> list[InstructionType]:
    """Phase-2 minimal preparation: drop barriers, preserve instruction order."""
    return [inst for inst in instructions if not isinstance(_unwrap_conditional(inst), BarrierInstruction)]


def _initial_segments(
    instructions: list[InstructionType],
    init_seg_size: int,
    n_qubits: int,
    nodes: int,
    max_hedge_dist: int,
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

        hyp = _build_hypergraph_from_instructions(this_insts, n_qubits)
        part = partition_hypergraph(hyp, n_qubits, nodes, DEFAULT_CONFIG_PATH)

        segments.append(
            Segment(
                gates=this_insts,
                hypergraph=hyp,
                partition=part,
                seam=SeamCompute(),
                segment_range=(seg_id, seg_id),
            )
        )
        seg_id += 1

    return segments


def _annotate_partitioned_circuit(segments: list[Segment]) -> PartitionedCircuit:
    """Add the appropriate telegate and teledata annotations to the partitioned segments and boundaries."""
    public_segments = _project_segments_to_swaps(segments)
    boundaries: list[SegmentBoundary] = []

    for idx, seg in enumerate(public_segments):
        if idx + 1 < len(public_segments):
            boundary = _build_boundary(seg, public_segments[idx + 1], idx)
            boundaries.append(boundary)

    return PartitionedCircuit(
        segments=public_segments,
        boundaries=boundaries,
    )


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


def _interaction_qubits(inst: InstructionType) -> list[int]:
    return list(getattr(_unwrap_conditional(inst), "qubits", []) or [])


def _target_qubit(inst: InstructionType) -> int | None:
    inner = _unwrap_conditional(inst)
    qubits = list(getattr(inner, "qubits", []) or [])
    if len(qubits) == 1:
        return qubits[0]
    qubit = getattr(inner, "qubit", None)
    return qubit if isinstance(qubit, int) else None


def _build_hypergraph_from_instructions(
    instructions: list[InstructionType], n_qubits: int
) -> Hypergraph:
    """Build hypergraph directly from bosonic instructions."""
    qubits = {qubit_id: QubitVertex(qubit_id=qubit_id) for qubit_id in range(n_qubits)}
    interactions: dict[int, InteractionVertex] = {}
    interaction_id = 0

    for position, inst in enumerate(instructions):
        if not _is_interaction(inst):
            continue
        inst_qubits = tuple(_interaction_qubits(inst))
        interactions[interaction_id] = InteractionVertex(
            interaction_id=interaction_id,
            position=position,
            qubits=inst_qubits,
        )
        interaction_id += 1

    return Hypergraph(qubits=qubits, interactions=interactions)


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


def _count_nonlocal_interactions(circuit: PartitionedCircuit) -> int:
    return sum(isinstance(op, NonlocalCZOp) for op in iter_annotated_operations(circuit))


def _count_interactions(instructions: Iterable[InstructionType]) -> int:
    return sum(1 for inst in instructions if _is_interaction(inst))


def _count_teleports(circuit: PartitionedCircuit) -> int:
    return sum(len(boundary.teleports) for boundary in circuit.boundaries)


def _count_swaps(circuit: PartitionedCircuit) -> int:
    return sum(len(boundary.swaps or []) for boundary in circuit.boundaries)


def _to_partitioned_segment(seg: Segment, idx: int) -> PartitionedSegment:
    return PartitionedSegment(
        segment_id=SegmentId(idx),
        instructions=seg.gates,
        partition={QubitId(w): NodeId(b) for w, b in seg.partition.items()},
    )


def _project_segments_to_swaps(segments: list[Segment]) -> list[PartitionedSegment]:
    if not segments:
        return []

    projected: list[PartitionedSegment] = [_to_partitioned_segment(segments[0], 0)]
    previous = dict(segments[0].partition)
    n_qubits = _partition_width(previous)
    for idx, seg in enumerate(segments[1:], 1):
        projection = project_boundary_swaps(previous, seg.partition, n_qubits)
        projected.append(
            PartitionedSegment(
                segment_id=SegmentId(idx),
                instructions=seg.gates,
                partition={QubitId(q): NodeId(node) for q, node in projection.partition.items()},
            )
        )
        previous = projection.partition
    return projected


def _build_boundary(
    left: PartitionedSegment, right: PartitionedSegment, boundary_idx: int
) -> SegmentBoundary:
    int_left = {int(q): int(node) for q, node in left.partition.items()}
    int_right = {int(q): int(node) for q, node in right.partition.items()}
    projection = project_boundary_swaps(int_left, int_right, _partition_width(int_left))
    swaps = [
        SwapBoundary(
            left_qubit=QubitId(swap.left_qubit),
            right_qubit=QubitId(swap.right_qubit),
            left_node=NodeId(swap.left_node),
            right_node=NodeId(swap.right_node),
        )
        for swap in projection.swaps
    ]
    return SegmentBoundary(
        boundary_id=BoundaryId(boundary_idx),
        left_segment_id=left.segment_id,
        right_segment_id=right.segment_id,
        teleports=[],
        swaps=swaps,
    )


def iter_annotated_operations(circuit: PartitionedCircuit) -> Iterator[AnnotatedOp]:
    if len(circuit.boundaries) != max(0, len(circuit.segments) - 1):
        raise ValueError("partitioned circuit must have exactly one boundary between adjacent segments")

    for idx, seg in enumerate(circuit.segments):
        yield from _annotate_segment_ops(seg)
        if idx < len(circuit.boundaries):
            boundary = circuit.boundaries[idx]
            expected_left = circuit.segments[idx].segment_id
            expected_right = circuit.segments[idx + 1].segment_id
            if boundary.left_segment_id != expected_left or boundary.right_segment_id != expected_right:
                raise ValueError("partitioned circuit boundaries must align with adjacent segment ordering")
            for swap in boundary.swaps or []:
                yield BoundarySwapOp(
                    boundary_id=boundary.boundary_id,
                    left_qubit=swap.left_qubit,
                    right_qubit=swap.right_qubit,
                    left_node=swap.left_node,
                    right_node=swap.right_node,
                )
            for teleport in boundary.teleports:
                yield BoundaryTeleportOp(
                    boundary_id=boundary.boundary_id,
                    qubit=teleport.qubit,
                    from_node=teleport.from_node,
                    to_node=teleport.to_node,
                )


def _annotate_segment_ops(seg: PartitionedSegment) -> Iterator[AnnotatedOp]:
    for inst in seg.instructions:
        inner = _unwrap_conditional(inst)
        qubits = tuple(_interaction_qubits(inst) if _is_interaction(inst) else (getattr(inner, "qubits", []) or []))
        nodes = tuple(seg.partition[QubitId(w)] for w in qubits if QubitId(w) in seg.partition)

        if isinstance(inner, CzInstruction):
            control_qubit = QubitId(inner.control)
            target_qubit = QubitId(inner.target)
            control_node = seg.partition[control_qubit]
            target_node = seg.partition[target_qubit]
            if control_node != target_node:
                yield NonlocalCZOp(
                    segment_id=seg.segment_id,
                    instruction=inst,
                    control_qubit=control_qubit,
                    target_qubit=target_qubit,
                    control_node=control_node,
                    target_node=target_node,
                )
                continue

        yield LocalOp(segment_id=seg.segment_id, instruction=inst, nodes=nodes)


def _partition_width(partition: dict[int, int]) -> int:
    return max(partition, default=-1) + 1
