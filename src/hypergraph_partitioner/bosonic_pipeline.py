"""Bosonic-model-native stats pipeline for partitioning metrics.

This module intentionally keeps the proven seam/merge heuristics while switching
runtime data flow to bosonic_model instructions.
"""

from __future__ import annotations

from collections.abc import Iterable

from bosonic_model import BarrierInstruction, Circuit, ConditionalInstruction
from bosonic_model.instructions import InstructionType

from hypergraph_partitioner.hgraph_builder import _split_long_hedges
from hypergraph_partitioner.models.hypergraph import Hedge, Hypergraph
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
    pos = 0
    cz_vertex = 0
    hyp: dict[int, list[Hedge]] = {}

    for inst in reversed(instructions):
        if _is_interaction(inst):
            wires = _interaction_wires(inst)
            for w in wires:
                if w not in hyp:
                    hyp[w] = [Hedge(nan=0, wires=[(cz_vertex - 1, pos)], out_pos=pos + 1)]
                else:
                    last = hyp[w][-1]
                    hyp[w][-1] = Hedge(
                        nan=last.nan,
                        wires=[(cz_vertex - 1, pos)] + last.wires,
                        out_pos=last.out_pos,
                    )
            cz_vertex -= 1
            pos += 1
            continue

        target = _target_wire(inst)
        if target is not None:
            if target not in hyp:
                hyp[target] = [Hedge(nan=0, wires=[], out_pos=pos)]
            else:
                last = hyp[target][-1]
                updated_last = Hedge(nan=0, wires=last.wires, out_pos=last.out_pos)
                hyp[target] = hyp[target][:-1] + [updated_last, Hedge(nan=0, wires=[], out_pos=pos)]
        pos += 1

    hyp = {w: _split_long_hedges(hedges, max_hedge_dist) for w, hedges in hyp.items()}
    return {w: [h for h in hedges if h.wires] for w, hedges in hyp.items()}


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
) -> list[Segment]:
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

    return merge_seams(to_hyp, to_part, k, n_qubits, initial)


def _preprocess(circuit: Circuit) -> Circuit:
    # step 1
    res = normalize_to_one_qubit_and_cz(circuit)
    # step 2 ...
    return res


def count_nonlocal_interactions(segments: list[Segment]) -> int:
    total = 0
    for seg in segments:
        for inst in seg.gates:
            if not _is_interaction(inst):
                continue
            blocks = {seg.partition.get(w) for w in _interaction_wires(inst) if seg.partition.get(w) is not None}
            total += max(0, len(blocks) - 1)
    return total


def count_interactions(instructions: Iterable[InstructionType]) -> int:
    return sum(1 for inst in instructions if _is_interaction(inst))


def count_teleports(segments: list[Segment], n_wires: int) -> int:
    total = 0
    for i in range(len(segments) - 1):
        left = segments[i].partition
        right = segments[i + 1].partition
        total += sum(1 for w in range(n_wires) if w in left and w in right and left[w] != right[w])
    return total
