"""Lower symbolic distributed operations into explicit protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

from bosonic_model import (
    Circuit,
    ConditionalInstruction,
    CzInstruction,
    DistributedCircuit,
    GateInstruction,
    InstructionType,
    RzzInstruction,
    SwapInstruction,
    UInstruction,
)

from hypergraph_partitioner.qpu_utils import (
    QpuLayout,
    alloc_cbit,
    append_instruction,
    append_shared_instruction,
    conditionally_wrap_instruction,
    finalize_circuit_registers,
    layouts_from_qubits_per_node,
    max_existing_cbit_in_distributed,
    node_for_qubit,
    single_node_for_qubits,
    total_qubits_from_map,
)
from .protocols import _emit_telegate_protocol, _emit_teledata_protocol
from .scheduler import RemoteOperationScheduler


@dataclass
class CircuitLoweringState:
    qpu_layouts: dict[int, QpuLayout]
    circuits: dict[int, Circuit]
    scheduler: RemoteOperationScheduler
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
        scheduler=RemoteOperationScheduler(qpu_layouts),
        next_cbit=max_existing_cbit_in_distributed(distributed) + 1,
    )

    monolithic = distributed.as_monolithic_circuit()
    for inst in monolithic.instructions:
        if _is_remote_cz(inst):
            _lower_remote_cz_instruction(inst, distributed.qubits_per_node, state)
        elif _is_remote_swap(inst):
            _lower_remote_swap_instruction(inst, distributed.qubits_per_node, state)
        elif _is_teleport(inst):
            _lower_teleport_instruction(inst, distributed.qubits_per_node, state)
        elif _is_local_cz(inst, distributed.qubits_per_node):
            _lower_local_cz_instruction(inst, distributed.qubits_per_node, state)
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
    operation = state.scheduler.begin((control_node, target_node))
    try:
        comm_ctrl = operation.alloc_comm(control_node, "remote_cz.control")
        comm_tgt = operation.alloc_comm(target_node, "remote_cz.target")
        c_start = alloc_cbit(state)
        c_end = alloc_cbit(state)

        remote_bell = GateInstruction(
            name="remote_link_phi_plus",
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
    finally:
        state.scheduler.finish(operation)


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
    operation = state.scheduler.begin((source_node, destination_node))
    try:
        comm_src = operation.alloc_comm(source_node, "teleport.source")
        c_data = alloc_cbit(state)
        c_comm = alloc_cbit(state)

        remote_bell = GateInstruction(
            name="remote_link_phi_plus",
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
    finally:
        state.scheduler.finish(operation)


def _lower_remote_swap_instruction(
    inst: InstructionType,
    qubits_per_node: dict[int, list[int]],
    state: CircuitLoweringState,
) -> None:
    outer_condition = inst.condition if isinstance(inst, ConditionalInstruction) else None
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    if not isinstance(inner, GateInstruction):
        raise TypeError(f"expected remote_swap gate, got {type(inner).__name__}")

    left_qubit, right_qubit = inner.qubits[:2]
    left_node = node_for_qubit(left_qubit, qubits_per_node)
    right_node = node_for_qubit(right_qubit, qubits_per_node)
    if left_node == right_node:
        append_instruction(
            state.circuits,
            state.instruction_index,
            left_node,
            conditionally_wrap_instruction(
                SwapInstruction(a=left_qubit, b=right_qubit, qubits=[left_qubit, right_qubit]),
                outer_condition,
            ),
            state,
        )
        return

    operation = state.scheduler.begin((left_node, right_node))
    try:
        left_comm = operation.alloc_comm(left_node, "remote_swap.first_source")
        right_recv = operation.alloc_comm(right_node, "remote_swap.first_destination")
        right_comm = operation.alloc_comm(right_node, "remote_swap.second_source")

        first_c_data = alloc_cbit(state)
        first_c_comm = alloc_cbit(state)
        second_c_data = alloc_cbit(state)
        second_c_comm = alloc_cbit(state)

        _emit_remote_teledata(
            data_src=left_qubit,
            comm_src=left_comm,
            comm_dst=right_recv,
            source_node=left_node,
            destination_node=right_node,
            c_data=first_c_data,
            c_comm=first_c_comm,
            outer_condition=outer_condition,
            state=state,
        )
        _emit_remote_teledata(
            data_src=right_qubit,
            comm_src=right_comm,
            comm_dst=left_qubit,
            source_node=right_node,
            destination_node=left_node,
            c_data=second_c_data,
            c_comm=second_c_comm,
            outer_condition=outer_condition,
            state=state,
        )

        local_swap = SwapInstruction(
            a=right_recv,
            b=right_qubit,
            qubits=[right_recv, right_qubit],
        )
        append_instruction(
            state.circuits,
            state.instruction_index,
            right_node,
            conditionally_wrap_instruction(local_swap, outer_condition),
            state,
        )
    finally:
        state.scheduler.finish(operation)


def _emit_remote_teledata(
    *,
    data_src: int,
    comm_src: int,
    comm_dst: int,
    source_node: int,
    destination_node: int,
    c_data: int,
    c_comm: int,
    outer_condition,
    state: CircuitLoweringState,
) -> None:
    remote_bell = GateInstruction(
        name="remote_link_phi_plus",
        qubits=[comm_src, comm_dst],
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
        data_src=data_src,
        comm_src=comm_src,
        comm_dst=comm_dst,
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


def _lower_local_cz_instruction(
    inst: InstructionType,
    qubits_per_node: dict[int, list[int]],
    state: CircuitLoweringState,
) -> None:
    outer_condition = inst.condition if isinstance(inst, ConditionalInstruction) else None
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    if not _is_cz_instruction(inner):
        raise TypeError(f"expected cz gate, got {type(inner).__name__}")

    control_qubit, target_qubit = list(inner.qubits[:2])
    node = single_node_for_qubits([control_qubit, target_qubit], qubits_per_node)
    for emitted in _emit_local_cz_decomposition(control_qubit, target_qubit):
        append_instruction(
            state.circuits,
            state.instruction_index,
            node,
            conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )


def _emit_local_cz_decomposition(control_qubit: int, target_qubit: int) -> list[InstructionType]:
    return [
        UInstruction(
            qubit=control_qubit,
            qubits=[control_qubit],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        ),
        UInstruction(
            qubit=target_qubit,
            qubits=[target_qubit],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        ),
        RzzInstruction(
            a=control_qubit,
            b=target_qubit,
            qubits=[control_qubit, target_qubit],
            theta=pi / 2,
            params=[pi / 2],
        ),
    ]


def _is_remote_cz(inst: InstructionType) -> bool:
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    return isinstance(inner, GateInstruction) and inner.name == "remote_cz"


def _is_teleport(inst: InstructionType) -> bool:
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    return isinstance(inner, GateInstruction) and inner.name == "teleport"


def _is_remote_swap(inst: InstructionType) -> bool:
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    return isinstance(inner, GateInstruction) and inner.name == "remote_swap"


def _is_local_cz(inst: InstructionType, qubits_per_node: dict[int, list[int]]) -> bool:
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    if not _is_cz_instruction(inner):
        return False
    try:
        single_node_for_qubits(inner.qubits, qubits_per_node)
    except ValueError:
        return False
    return True


def _is_cz_instruction(inst: InstructionType) -> bool:
    return isinstance(inst, CzInstruction) or (
        isinstance(inst, GateInstruction) and inst.name == "cz"
    )
