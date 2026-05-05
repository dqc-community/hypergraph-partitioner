"""Shared QPU layout and instruction utilities for distribution and lowering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from bosonic_model import (
    Circuit,
    Condition,
    ConditionalInstruction,
    DistributedCircuit,
    InstructionType,
    MeasureInstruction,
    Register,
)

from hypergraph_partitioner.bosonic_pipeline import iter_annotated_operations
from hypergraph_partitioner.models.circuit_annotations import PartitionedCircuit


@dataclass
class QpuLayout:
    node: int
    data_slots: list[int]
    comm_slots: list[int]
    receiver_slots: list[int]
    free_comm: set[int] = field(default_factory=set)
    free_receiver: set[int] = field(default_factory=set)


class _CounterState(Protocol):
    next_cbit: int


class _OrderedState(Protocol):
    next_order: int


def num_nodes(partitioned: PartitionedCircuit) -> int:
    max_node = -1
    for seg in partitioned.segments:
        for node in seg.partition.values():
            max_node = max(max_node, int(node))
    return max_node + 1 if max_node >= 0 else 0


def validate_capacity(partitioned: PartitionedCircuit, qubits_per_node: int) -> None:
    for seg in partitioned.segments:
        per_node: dict[int, int] = {}
        for node in seg.partition.values():
            per_node[int(node)] = per_node.get(int(node), 0) + 1
        for node, count in per_node.items():
            if count > qubits_per_node:
                raise ValueError(
                    f"segment {seg.segment_id} assigns {count} qubits to node {node}, "
                    f"exceeding qubits_per_node={qubits_per_node}"
                )


def max_existing_cbit(partitioned: PartitionedCircuit) -> int:
    maximum = -1
    for op in iter_annotated_operations(partitioned):
        inst = getattr(op, "instruction", None)
        if inst is not None:
            maximum = max(maximum, max_cbit_in_instruction(inst))
    return maximum


def max_existing_cbit_in_distributed(distributed: DistributedCircuit) -> int:
    maximum = -1
    for circuit in distributed.circuits.values():
        for inst in circuit.instructions:
            maximum = max(maximum, max_cbit_in_instruction(inst))
    return maximum


def max_cbit_in_instruction(inst: InstructionType) -> int:
    if isinstance(inst, MeasureInstruction):
        return inst.cbit
    if isinstance(inst, ConditionalInstruction):
        return max(_condition_cbit(inst.condition), max_cbit_in_instruction(inst.op))
    return -1


def _condition_cbit(condition: Condition) -> int:
    return getattr(condition, "cbit", getattr(condition, "creg_base", 0))


def _condition_value(condition: Condition) -> bool:
    return getattr(condition, "value", bool(getattr(condition, "creg_value", 0)))


def make_condition(cbit: int, value: bool) -> Condition:
    fields = getattr(Condition, "model_fields", {})
    if "cbit" in fields:
        return Condition(cbit=cbit, value=value)
    return Condition(creg_base=cbit, creg_size=1, creg_value=int(value))


def _remap_condition(condition: Condition, cbit_map: dict[int, int]) -> Condition:
    condition_cbit = _condition_cbit(condition)
    mapped_cbit = cbit_map.get(condition_cbit, condition_cbit)
    fields = getattr(Condition, "model_fields", {})
    if "cbit" in fields:
        return Condition(cbit=mapped_cbit, value=_condition_value(condition))
    return condition.model_copy(update={"creg_base": mapped_cbit})


def build_qpu_layouts(qubits_per_node: int, n_nodes: int) -> dict[int, QpuLayout]:
    qpu_layouts: dict[int, QpuLayout] = {}
    physical_slots_per_node = qubits_per_node + 2
    for node in range(n_nodes):
        base = node * physical_slots_per_node
        data_slots = list(range(base, base + qubits_per_node))
        comm_slots = list(range(base + qubits_per_node, base + qubits_per_node + 2))
        receiver_slots: list[int] = []
        qpu_layouts[node] = QpuLayout(
            node=node,
            data_slots=data_slots,
            comm_slots=comm_slots,
            receiver_slots=receiver_slots,
            free_comm=set(comm_slots),
            free_receiver=set(receiver_slots),
        )
    return qpu_layouts


def layouts_from_qubits_per_node(qubits_per_node: dict[int, list[int]]) -> dict[int, QpuLayout]:
    layouts: dict[int, QpuLayout] = {}
    for node, qubits in qubits_per_node.items():
        if len(qubits) < 3:
            raise ValueError(
                f"node {node} has {len(qubits)} qubits; expected at least one data qubit and two aux qubits"
            )
        data_slots = list(qubits[:-2])
        comm_slots = list(qubits[-2:])
        receiver_slots: list[int] = []
        layouts[node] = QpuLayout(
            node=node,
            data_slots=data_slots,
            comm_slots=comm_slots,
            receiver_slots=receiver_slots,
            free_comm=set(comm_slots),
            free_receiver=set(receiver_slots),
        )
    return layouts


def alloc_comm(layout: QpuLayout) -> int:
    if not layout.free_comm:
        raise ValueError(f"node {layout.node} has no free communication qubits")
    qubit = min(layout.free_comm)
    layout.free_comm.remove(qubit)
    return qubit


def free_comm(layout: QpuLayout, qubit: int) -> None:
    layout.free_comm.add(qubit)


def alloc_receiver(layout: QpuLayout) -> int:
    if not layout.free_receiver:
        raise ValueError(f"node {layout.node} has no free receiver qubits")
    qubit = min(layout.free_receiver)
    layout.free_receiver.remove(qubit)
    return qubit


def free_receiver(layout: QpuLayout, qubit: int) -> None:
    layout.free_receiver.add(qubit)


def alloc_cbit(state: _CounterState) -> int:
    cbit = state.next_cbit
    state.next_cbit += 1
    return cbit


def append_instruction(
    circuits: dict[int, Circuit],
    instruction_index: dict[int, int],
    node: int,
    inst: InstructionType,
    state: _OrderedState,
) -> None:
    circuits[node].instructions.append(inst)
    instruction_index[id(inst)] = state.next_order
    state.next_order += 1


def append_shared_instruction(
    circuits: dict[int, Circuit],
    instruction_index: dict[int, int],
    nodes: tuple[int, int],
    inst: InstructionType,
    state: _OrderedState,
) -> None:
    for node in nodes:
        circuits[node].instructions.append(inst)
    instruction_index[id(inst)] = state.next_order
    state.next_order += 1


def node_for_qubit(qubit: int, qubits_per_node: dict[int, list[int]]) -> int:
    for node, assigned in qubits_per_node.items():
        if qubit in assigned:
            return node
    raise ValueError(f"Qubit {qubit} is not assigned to any node")


def single_node_for_qubits(qubits: list[int], qubits_per_node: dict[int, list[int]]) -> int:
    nodes = {node_for_qubit(qubit, qubits_per_node) for qubit in qubits}
    if len(nodes) != 1:
        raise ValueError(f"expected local instruction, got qubits spanning nodes: {qubits}")
    return next(iter(nodes))


def total_qubits_from_map(qubits_per_node: dict[int, list[int]]) -> int:
    max_qubit = -1
    for qubits in qubits_per_node.values():
        if qubits:
            max_qubit = max(max_qubit, max(qubits))
    return max_qubit + 1 if max_qubit >= 0 else 0


def finalize_circuit_registers(
    circuits: dict[int, Circuit], *, total_qubits: int, total_cbits: int
) -> None:
    qregs = {"q": Register(name="q", size=total_qubits, base=0)} if total_qubits else {}
    cregs = {"c": Register(name="c", size=total_cbits, base=0)} if total_cbits else {}
    for circuit in circuits.values():
        circuit.qregs = qregs
        circuit.cregs = cregs


def conditionally_wrap_instruction(
    inst: InstructionType, condition: Condition | None
) -> InstructionType:
    if condition is None:
        return inst
    return ConditionalInstruction(
        condition=condition,
        op=inst,
        qubits=list(inst.qubits),
    )


def remap_instruction(
    inst: InstructionType,
    qubit_map: dict[int, int],
    cbit_map: dict[int, int] | None = None,
) -> InstructionType:
    cbit_map = cbit_map or {}
    if isinstance(inst, ConditionalInstruction):
        mapped_op = remap_instruction(inst.op, qubit_map, cbit_map)
        return inst.model_copy(
            update={
                "qubits": [qubit_map.get(q, q) for q in inst.qubits],
                "condition": _remap_condition(inst.condition, cbit_map),
                "op": mapped_op,
            }
        )

    update: dict[str, object] = {"qubits": [qubit_map.get(q, q) for q in inst.qubits]}
    for field_name in (
        "qubit",
        "control",
        "target",
        "a",
        "b",
        "control1",
        "control2",
        "control3",
        "control4",
        "target1",
        "target2",
    ):
        if hasattr(inst, field_name):
            value = getattr(inst, field_name)
            update[field_name] = qubit_map.get(value, value)
    if isinstance(inst, MeasureInstruction):
        update["cbit"] = cbit_map.get(inst.cbit, inst.cbit)
    return inst.model_copy(update=update)
