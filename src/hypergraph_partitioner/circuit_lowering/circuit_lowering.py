"""Lower symbolic distributed operations into explicit protocols."""

from __future__ import annotations

from dataclasses import dataclass, field

from bosonic_model import Circuit, ConditionalInstruction, DistributedCircuit, GateInstruction, InstructionType

from hypergraph_partitioner.qpu_utils import (
    QpuLayout,
    alloc_cbit,
    alloc_comm,
    append_instruction,
    append_shared_instruction,
    conditionally_wrap_instruction,
    finalize_circuit_registers,
    free_comm,
    layouts_from_qubits_per_node,
    max_existing_cbit_in_distributed,
    node_for_qubit,
    single_node_for_qubits,
    total_qubits_from_map,
)
from .protocols import _emit_telegate_protocol, _emit_teledata_protocol


@dataclass
class CircuitLoweringState:
    qpu_layouts: dict[int, QpuLayout]
    circuits: dict[int, Circuit]
    instruction_index: dict[int, int] = field(default_factory=dict)
    next_order: int = 0
    next_cbit: int = 0


def lower_distributed_circuit(distributed: DistributedCircuit) -> DistributedCircuit:
    if not distributed.circuits:
        return distributed

    qpu_layouts = layouts_from_qubits_per_node(distributed.qubits_per_node)
    state = CircuitLoweringState(
        qpu_layouts=qpu_layouts,
        circuits={node: Circuit() for node in distributed.circuits},
        next_cbit=max_existing_cbit_in_distributed(distributed) + 1,
    )

    monolithic = distributed.as_monolithic_circuit()
    for inst in monolithic.instructions:
        if _is_remote_cz(inst):
            _lower_remote_cz_instruction(inst, distributed.qubits_per_node, state)
        elif _is_teleport(inst):
            _lower_teleport_instruction(inst, distributed.qubits_per_node, state)
        else:
            node = single_node_for_qubits(inst.qubits, distributed.qubits_per_node)
            append_instruction(state.circuits, state.instruction_index, node, inst, state)

    total_qubits = total_qubits_from_map(distributed.qubits_per_node)
    finalize_circuit_registers(state.circuits, total_qubits=total_qubits, total_cbits=state.next_cbit)
    lowered = DistributedCircuit(
        qubits_per_node=distributed.qubits_per_node,
        circuits=state.circuits,
    )
    lowered._instruction_index = state.instruction_index
    return lowered


def _lower_remote_cz_instruction(
    inst: InstructionType,
    qubits_per_node: dict[int, list[int]],
    state: CircuitLoweringState,
) -> None:
    outer_condition = inst.condition if isinstance(inst, ConditionalInstruction) else None
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    if not isinstance(inner, GateInstruction):
        raise TypeError(f"expected remote_cz gate, got {type(inner).__name__}")

    control_qubit, target_qubit = inner.qubits[:2]
    control_node = node_for_qubit(control_qubit, qubits_per_node)
    target_node = node_for_qubit(target_qubit, qubits_per_node)
    control_layout = state.qpu_layouts[control_node]
    target_layout = state.qpu_layouts[target_node]
    comm_ctrl = alloc_comm(control_layout)
    comm_tgt = alloc_comm(target_layout)
    c_start = alloc_cbit(state)
    c_end = alloc_cbit(state)

    remote_bell = GateInstruction(
        name="bell_pair_phi_plus",
        qubits=[comm_ctrl, comm_tgt],
        params=[],
        opaque=True,
    )
    append_shared_instruction(
        state.circuits,
        state.instruction_index,
        (control_node, target_node),
        conditionally_wrap_instruction(remote_bell, outer_condition),
        state,
    )

    source_prefix: list[InstructionType] = []
    target_body: list[InstructionType] = []
    source_suffix: list[InstructionType] = []
    _emit_telegate_protocol(
        source_prefix,
        data_ctrl=control_qubit,
        comm_ctrl=comm_ctrl,
        comm_tgt=comm_tgt,
        data_tgt=target_qubit,
        c_start=c_start,
        c_end=c_end,
        include_bell=False,
        target_instructions=target_body,
        source_final_instructions=source_suffix,
    )
    for emitted in source_prefix:
        append_instruction(
            state.circuits,
            state.instruction_index,
            control_node,
            conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )
    for emitted in target_body:
        append_instruction(
            state.circuits,
            state.instruction_index,
            target_node,
            conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )
    for emitted in source_suffix:
        append_instruction(
            state.circuits,
            state.instruction_index,
            control_node,
            conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )

    free_comm(control_layout, comm_ctrl)
    free_comm(target_layout, comm_tgt)


def _lower_teleport_instruction(
    inst: InstructionType,
    qubits_per_node: dict[int, list[int]],
    state: CircuitLoweringState,
) -> None:
    outer_condition = inst.condition if isinstance(inst, ConditionalInstruction) else None
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    if not isinstance(inner, GateInstruction):
        raise TypeError(f"expected teleport gate, got {type(inner).__name__}")

    source_qubit, destination_qubit = inner.qubits[:2]
    source_node = node_for_qubit(source_qubit, qubits_per_node)
    destination_node = node_for_qubit(destination_qubit, qubits_per_node)
    source_layout = state.qpu_layouts[source_node]
    comm_src = alloc_comm(source_layout)
    c_data = alloc_cbit(state)
    c_comm = alloc_cbit(state)

    remote_bell = GateInstruction(
        name="bell_pair_phi_plus",
        qubits=[comm_src, destination_qubit],
        params=[],
        opaque=True,
    )
    append_shared_instruction(
        state.circuits,
        state.instruction_index,
        (source_node, destination_node),
        conditionally_wrap_instruction(remote_bell, outer_condition),
        state,
    )

    source_body: list[InstructionType] = []
    destination_body: list[InstructionType] = []
    _emit_teledata_protocol(
        source_body,
        data_src=source_qubit,
        comm_src=comm_src,
        comm_dst=destination_qubit,
        c_data=c_data,
        c_comm=c_comm,
        include_bell=False,
        dest_instructions=destination_body,
    )
    for emitted in source_body:
        append_instruction(
            state.circuits,
            state.instruction_index,
            source_node,
            conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )
    for emitted in destination_body:
        append_instruction(
            state.circuits,
            state.instruction_index,
            destination_node,
            conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )

    free_comm(source_layout, comm_src)


def _is_remote_cz(inst: InstructionType) -> bool:
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    return isinstance(inner, GateInstruction) and inner.name == "remote_cz"


def _is_teleport(inst: InstructionType) -> bool:
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    return isinstance(inner, GateInstruction) and inner.name == "teleport"
