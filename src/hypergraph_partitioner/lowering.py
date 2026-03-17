"""Protocol-building and lowering helpers for telegate and teledata."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

from bosonic_model import (
    Circuit,
    Condition,
    ConditionalInstruction,
    DistributedCircuit,
    GateInstruction,
    InstructionType,
    MeasureInstruction,
    Register,
    ResetInstruction,
    RzzInstruction,
    UInstruction,
)
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from hypergraph_partitioner.bosonic_pipeline import iter_annotated_operations
from hypergraph_partitioner.models.circuit_annotations import (
    BoundaryTeleportOp,
    LocalOp,
    NonlocalCZOp,
    PartitionedCircuit,
)


@dataclass(frozen=True)
class PhysicalLocation:
    node: int
    qubit: int


@dataclass
class QpuLayout:
    node: int
    data_slots: list[int]
    comm_slots: list[int]
    receiver_slots: list[int]
    free_comm: set[int] = field(default_factory=set)
    free_receiver: set[int] = field(default_factory=set)


@dataclass
class DistributionState:
    qpu_layouts: dict[int, QpuLayout]
    circuits: dict[int, Circuit]
    qubit_locations: dict[int, PhysicalLocation]
    instruction_index: dict[int, int] = field(default_factory=dict)
    next_order: int = 0
    next_cbit: int = 0


@dataclass
class ProtocolLoweringState:
    qpu_layouts: dict[int, QpuLayout]
    circuits: dict[int, Circuit]
    instruction_index: dict[int, int] = field(default_factory=dict)
    next_order: int = 0
    next_cbit: int = 0


def _registers(
    *, n_qubits: int, classical: tuple[tuple[str, int], ...]
) -> tuple[dict[str, Register], dict[str, Register]]:
    qregs = {"q": Register(name="q", size=n_qubits, base=0)}
    cregs: dict[str, Register] = {}
    base = 0
    for name, size in classical:
        cregs[name] = Register(name=name, size=size, base=base)
        base += size
    return qregs, cregs


def bell_pair_phi_plus_matrix() -> list[list[complex]]:
    scale = 1 / (2**0.5)
    return [
        [1 * scale, 0, 1 * scale, 0],
        [0, 1 * scale, 0, 1 * scale],
        [0, 1 * scale, 0, -1 * scale],
        [1 * scale, 0, -1 * scale, 0],
    ]


def build_telegate_remote_cz_dsl() -> Circuit:
    qregs, cregs = _registers(n_qubits=4, classical=(("c_start", 1), ("c_end", 1)))
    instructions: list[InstructionType] = []
    _emit_telegate_protocol(
        instructions,
        data_ctrl=0,
        comm_ctrl=1,
        comm_tgt=2,
        data_tgt=3,
        c_start=0,
        c_end=1,
    )
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def build_ideal_remote_cz_dsl() -> Circuit:
    qregs, cregs = _registers(n_qubits=4, classical=())
    instructions: list[InstructionType] = [
        GateInstruction(name="cz", qubits=[0, 3], params=[], opaque=True)
    ]
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def build_teledata_dsl() -> Circuit:
    qregs, cregs = _registers(n_qubits=3, classical=(("c_data", 1), ("c_comm", 1)))
    instructions: list[InstructionType] = []
    _emit_teledata_protocol(instructions, data_src=0, comm_src=1, comm_dst=2, c_data=0, c_comm=1)
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def build_ideal_teledata_dsl() -> Circuit:
    qregs, cregs = _registers(n_qubits=3, classical=())
    return Circuit(qregs=qregs, cregs=cregs, instructions=[])


def annotated_to_distributed_circuit(
    partitioned: PartitionedCircuit, *, qpu_data_capacity: int
) -> DistributedCircuit:
    if qpu_data_capacity < 1:
        raise ValueError("qpu_data_capacity must be positive")
    if not partitioned.segments:
        return DistributedCircuit(qubits_per_node={}, circuits={})

    n_nodes = _num_nodes(partitioned)
    _validate_capacity(partitioned, qpu_data_capacity)
    state = _initialize_distribution_state(partitioned, qpu_data_capacity, n_nodes)

    for op in iter_annotated_operations(partitioned):
        if isinstance(op, LocalOp):
            _distribute_local(op, state)
        elif isinstance(op, NonlocalCZOp):
            _distribute_telegate(op, state)
        elif isinstance(op, BoundaryTeleportOp):
            _distribute_teledata(op, state)
        else:
            raise TypeError(f"unsupported annotated op: {type(op).__name__}")

    _finalize_circuit_registers(
        state.circuits,
        total_qubits=n_nodes * 3 * qpu_data_capacity,
        total_cbits=state.next_cbit,
    )
    distributed = DistributedCircuit(
        qubits_per_node={
            node: layout.data_slots + layout.comm_slots + layout.receiver_slots
            for node, layout in state.qpu_layouts.items()
        },
        circuits=state.circuits,
    )
    distributed._instruction_index = state.instruction_index
    return distributed


def lower_distributed_circuit(distributed: DistributedCircuit) -> DistributedCircuit:
    if not distributed.circuits:
        return distributed

    qpu_layouts = _layouts_from_qubits_per_node(distributed.qubits_per_node)
    state = ProtocolLoweringState(
        qpu_layouts=qpu_layouts,
        circuits={node: Circuit() for node in distributed.circuits},
        next_cbit=_max_existing_cbit_in_distributed(distributed) + 1,
    )

    monolithic = distributed.as_monolithic_circuit()
    for inst in monolithic.instructions:
        if _is_remote_cz(inst):
            _lower_remote_cz_instruction(inst, distributed.qubits_per_node, state)
        elif _is_teleport(inst):
            _lower_teleport_instruction(inst, distributed.qubits_per_node, state)
        else:
            node = _single_node_for_qubits(inst.qubits, distributed.qubits_per_node)
            _append_instruction(state.circuits, state.instruction_index, node, inst, state)

    total_qubits = _total_qubits_from_map(distributed.qubits_per_node)
    _finalize_circuit_registers(state.circuits, total_qubits=total_qubits, total_cbits=state.next_cbit)
    lowered = DistributedCircuit(
        qubits_per_node=distributed.qubits_per_node,
        circuits=state.circuits,
    )
    lowered._instruction_index = state.instruction_index
    return lowered


def to_aer_compatible_qiskit(circuit: QuantumCircuit) -> QuantumCircuit:
    rewritten = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for inst in circuit.data:
        op = inst.operation
        if op.name in {"bell_pair_phi_plus", "remote_bell_pair_phi_plus"}:
            rewritten.append(
                UnitaryGate(bell_pair_phi_plus_matrix(), label=op.name),
                inst.qubits,
                inst.clbits,
            )
            continue
        if op.name in {"remote_link_psi_minus", "remote_link_psi_plus"}:
            rewritten.append(UnitaryGate(op.to_matrix(), label=op.name), inst.qubits, inst.clbits)
            continue
        rewritten.append(op, inst.qubits, inst.clbits)
    return rewritten


def _num_nodes(partitioned: PartitionedCircuit) -> int:
    max_node = -1
    for seg in partitioned.segments:
        for node in seg.partition.values():
            max_node = max(max_node, int(node))
    return max_node + 1 if max_node >= 0 else 0


def _validate_capacity(partitioned: PartitionedCircuit, qpu_data_capacity: int) -> None:
    for seg in partitioned.segments:
        per_node: dict[int, int] = {}
        for node in seg.partition.values():
            per_node[int(node)] = per_node.get(int(node), 0) + 1
        for node, count in per_node.items():
            if count > qpu_data_capacity:
                raise ValueError(
                    f"segment {seg.segment_id} assigns {count} qubits to node {node}, "
                    f"exceeding qpu_data_capacity={qpu_data_capacity}"
                )


def _max_existing_cbit(partitioned: PartitionedCircuit) -> int:
    maximum = -1
    for op in iter_annotated_operations(partitioned):
        if isinstance(op, LocalOp):
            maximum = max(maximum, _max_cbit_in_instruction(op.instruction))
    return maximum


def _max_existing_cbit_in_distributed(distributed: DistributedCircuit) -> int:
    maximum = -1
    for circuit in distributed.circuits.values():
        for inst in circuit.instructions:
            maximum = max(maximum, _max_cbit_in_instruction(inst))
    return maximum


def _max_cbit_in_instruction(inst: InstructionType) -> int:
    if isinstance(inst, MeasureInstruction):
        return inst.cbit
    if isinstance(inst, ConditionalInstruction):
        return max(inst.condition.cbit, _max_cbit_in_instruction(inst.op))
    return -1


def _initialize_distribution_state(
    partitioned: PartitionedCircuit, qpu_data_capacity: int, n_nodes: int
) -> DistributionState:
    qpu_layouts = _build_qpu_layouts(qpu_data_capacity, n_nodes)
    circuits = {node: Circuit() for node in range(n_nodes)}

    first_segment = partitioned.segments[0]
    qubit_locations: dict[int, PhysicalLocation] = {}
    for node in range(n_nodes):
        node_qubits = sorted(
            int(q) for q, owner in first_segment.partition.items() if int(owner) == node
        )
        for slot, q in zip(qpu_layouts[node].data_slots, node_qubits, strict=False):
            qubit_locations[q] = PhysicalLocation(node=node, qubit=slot)

    return DistributionState(
        qpu_layouts=qpu_layouts,
        circuits=circuits,
        qubit_locations=qubit_locations,
        next_cbit=_max_existing_cbit(partitioned) + 1,
    )


def _build_qpu_layouts(qpu_data_capacity: int, n_nodes: int) -> dict[int, QpuLayout]:
    qpu_layouts: dict[int, QpuLayout] = {}
    for node in range(n_nodes):
        base = node * 3 * qpu_data_capacity
        data_slots = list(range(base, base + qpu_data_capacity))
        comm_slots = list(range(base + qpu_data_capacity, base + 2 * qpu_data_capacity))
        receiver_slots = list(range(base + 2 * qpu_data_capacity, base + 3 * qpu_data_capacity))
        qpu_layouts[node] = QpuLayout(
            node=node,
            data_slots=data_slots,
            comm_slots=comm_slots,
            receiver_slots=receiver_slots,
            free_comm=set(comm_slots),
            free_receiver=set(receiver_slots),
        )
    return qpu_layouts


def _layouts_from_qubits_per_node(qubits_per_node: dict[int, list[int]]) -> dict[int, QpuLayout]:
    layouts: dict[int, QpuLayout] = {}
    for node, qubits in qubits_per_node.items():
        if len(qubits) % 3 != 0:
            raise ValueError(
                f"node {node} has {len(qubits)} qubits; expected a multiple of 3 for data/comm/receiver layout"
            )
        capacity = len(qubits) // 3
        data_slots = list(qubits[:capacity])
        comm_slots = list(qubits[capacity : 2 * capacity])
        receiver_slots = list(qubits[2 * capacity :])
        layouts[node] = QpuLayout(
            node=node,
            data_slots=data_slots,
            comm_slots=comm_slots,
            receiver_slots=receiver_slots,
            free_comm=set(comm_slots),
            free_receiver=set(receiver_slots),
        )
    return layouts


def _alloc_comm(layout: QpuLayout) -> int:
    if not layout.free_comm:
        raise ValueError(f"node {layout.node} has no free communication qubits")
    qubit = min(layout.free_comm)
    layout.free_comm.remove(qubit)
    return qubit


def _free_comm(layout: QpuLayout, qubit: int) -> None:
    layout.free_comm.add(qubit)


def _alloc_receiver(layout: QpuLayout) -> int:
    if not layout.free_receiver:
        raise ValueError(f"node {layout.node} has no free receiver qubits")
    qubit = min(layout.free_receiver)
    layout.free_receiver.remove(qubit)
    return qubit


def _alloc_cbit(state: DistributionState | ProtocolLoweringState) -> int:
    cbit = state.next_cbit
    state.next_cbit += 1
    return cbit


def _append_instruction(
    circuits: dict[int, Circuit],
    instruction_index: dict[int, int],
    node: int,
    inst: InstructionType,
    state: DistributionState | ProtocolLoweringState,
) -> None:
    circuits[node].instructions.append(inst)
    instruction_index[id(inst)] = state.next_order
    state.next_order += 1


def _append_shared_instruction(
    circuits: dict[int, Circuit],
    instruction_index: dict[int, int],
    nodes: tuple[int, int],
    inst: InstructionType,
    state: DistributionState | ProtocolLoweringState,
) -> None:
    for node in nodes:
        circuits[node].instructions.append(inst)
    instruction_index[id(inst)] = state.next_order
    state.next_order += 1


def _distribute_local(op: LocalOp, state: DistributionState) -> None:
    inst = op.instruction
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    qubits = list(getattr(inner, "qubits", []) or [])
    qubit_map = {
        qubit: state.qubit_locations[qubit].qubit
        for qubit in qubits
        if qubit in state.qubit_locations
    }
    mapped = _remap_instruction(inst, qubit_map)
    node = state.qubit_locations[qubits[0]].node if qubits else 0
    _append_instruction(state.circuits, state.instruction_index, node, mapped, state)


def _distribute_telegate(op: NonlocalCZOp, state: DistributionState) -> None:
    control = state.qubit_locations[int(op.control_qubit)]
    target = state.qubit_locations[int(op.target_qubit)]
    remote_cz = GateInstruction(
        name="remote_cz",
        qubits=[control.qubit, target.qubit],
        params=[],
        opaque=True,
    )
    if isinstance(op.instruction, ConditionalInstruction):
        inst = ConditionalInstruction(
            condition=op.instruction.condition,
            op=remote_cz,
            qubits=list(remote_cz.qubits),
        )
    else:
        inst = remote_cz
    _append_shared_instruction(
        state.circuits,
        state.instruction_index,
        (control.node, target.node),
        inst,
        state,
    )


def _distribute_teledata(op: BoundaryTeleportOp, state: DistributionState) -> None:
    qubit = int(op.qubit)
    source = state.qubit_locations[qubit]
    dst_layout = state.qpu_layouts[int(op.to_node)]
    dst_qubit = _alloc_receiver(dst_layout)
    inst = GateInstruction(name="teleport", qubits=[source.qubit, dst_qubit], params=[], opaque=True)
    _append_shared_instruction(
        state.circuits,
        state.instruction_index,
        (source.node, int(op.to_node)),
        inst,
        state,
    )
    state.qubit_locations[qubit] = PhysicalLocation(node=int(op.to_node), qubit=dst_qubit)


def _lower_remote_cz_instruction(
    inst: InstructionType,
    qubits_per_node: dict[int, list[int]],
    state: ProtocolLoweringState,
) -> None:
    outer_condition = inst.condition if isinstance(inst, ConditionalInstruction) else None
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    if not isinstance(inner, GateInstruction):
        raise TypeError(f"expected remote_cz gate, got {type(inner).__name__}")

    control_qubit, target_qubit = inner.qubits[:2]
    control_node = _node_for_qubit(control_qubit, qubits_per_node)
    target_node = _node_for_qubit(target_qubit, qubits_per_node)
    control_layout = state.qpu_layouts[control_node]
    target_layout = state.qpu_layouts[target_node]
    comm_ctrl = _alloc_comm(control_layout)
    comm_tgt = _alloc_comm(target_layout)
    c_start = _alloc_cbit(state)
    c_end = _alloc_cbit(state)

    remote_bell = GateInstruction(
        name="remote_bell_pair_phi_plus",
        qubits=[comm_ctrl, comm_tgt],
        params=[],
        opaque=True,
    )
    _append_shared_instruction(
        state.circuits,
        state.instruction_index,
        (control_node, target_node),
        _conditionally_wrap_instruction(remote_bell, outer_condition),
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
        _append_instruction(
            state.circuits,
            state.instruction_index,
            control_node,
            _conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )
    for emitted in target_body:
        _append_instruction(
            state.circuits,
            state.instruction_index,
            target_node,
            _conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )
    for emitted in source_suffix:
        _append_instruction(
            state.circuits,
            state.instruction_index,
            control_node,
            _conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )

    _free_comm(control_layout, comm_ctrl)
    _free_comm(target_layout, comm_tgt)


def _lower_teleport_instruction(
    inst: InstructionType,
    qubits_per_node: dict[int, list[int]],
    state: ProtocolLoweringState,
) -> None:
    outer_condition = inst.condition if isinstance(inst, ConditionalInstruction) else None
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    if not isinstance(inner, GateInstruction):
        raise TypeError(f"expected teleport gate, got {type(inner).__name__}")

    source_qubit, destination_qubit = inner.qubits[:2]
    source_node = _node_for_qubit(source_qubit, qubits_per_node)
    destination_node = _node_for_qubit(destination_qubit, qubits_per_node)
    source_layout = state.qpu_layouts[source_node]
    comm_src = _alloc_comm(source_layout)
    c_data = _alloc_cbit(state)
    c_comm = _alloc_cbit(state)

    remote_bell = GateInstruction(
        name="remote_bell_pair_phi_plus",
        qubits=[comm_src, destination_qubit],
        params=[],
        opaque=True,
    )
    _append_shared_instruction(
        state.circuits,
        state.instruction_index,
        (source_node, destination_node),
        _conditionally_wrap_instruction(remote_bell, outer_condition),
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
        _append_instruction(
            state.circuits,
            state.instruction_index,
            source_node,
            _conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )
    for emitted in destination_body:
        _append_instruction(
            state.circuits,
            state.instruction_index,
            destination_node,
            _conditionally_wrap_instruction(emitted, outer_condition),
            state,
        )

    _free_comm(source_layout, comm_src)


def _node_for_qubit(qubit: int, qubits_per_node: dict[int, list[int]]) -> int:
    for node, assigned in qubits_per_node.items():
        if qubit in assigned:
            return node
    raise ValueError(f"Qubit {qubit} is not assigned to any node")


def _single_node_for_qubits(qubits: list[int], qubits_per_node: dict[int, list[int]]) -> int:
    nodes = {_node_for_qubit(qubit, qubits_per_node) for qubit in qubits}
    if len(nodes) != 1:
        raise ValueError(f"expected local instruction, got qubits spanning nodes: {qubits}")
    return next(iter(nodes))


def _total_qubits_from_map(qubits_per_node: dict[int, list[int]]) -> int:
    max_qubit = -1
    for qubits in qubits_per_node.values():
        if qubits:
            max_qubit = max(max_qubit, max(qubits))
    return max_qubit + 1 if max_qubit >= 0 else 0


def _finalize_circuit_registers(
    circuits: dict[int, Circuit], *, total_qubits: int, total_cbits: int
) -> None:
    qregs = {"q": Register(name="q", size=total_qubits, base=0)} if total_qubits else {}
    cregs = {"c": Register(name="c", size=total_cbits, base=0)} if total_cbits else {}
    for circuit in circuits.values():
        circuit.qregs = qregs
        circuit.cregs = cregs


def _is_remote_cz(inst: InstructionType) -> bool:
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    return isinstance(inner, GateInstruction) and inner.name == "remote_cz"


def _is_teleport(inst: InstructionType) -> bool:
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    return isinstance(inner, GateInstruction) and inner.name == "teleport"


def _conditionally_wrap_instruction(
    inst: InstructionType, condition: Condition | None
) -> InstructionType:
    if condition is None:
        return inst
    return ConditionalInstruction(
        condition=condition,
        op=inst,
        qubits=list(inst.qubits),
    )


def _remap_instruction(
    inst: InstructionType,
    qubit_map: dict[int, int],
    cbit_map: dict[int, int] | None = None,
) -> InstructionType:
    cbit_map = cbit_map or {}
    if isinstance(inst, ConditionalInstruction):
        mapped_op = _remap_instruction(inst.op, qubit_map, cbit_map)
        mapped_cbit = cbit_map.get(inst.condition.cbit, inst.condition.cbit)
        return inst.model_copy(
            update={
                "qubits": [qubit_map.get(q, q) for q in inst.qubits],
                "condition": Condition(cbit=mapped_cbit, value=inst.condition.value),
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


def _emit_telegate_protocol(
    instructions: list[InstructionType],
    *,
    data_ctrl: int,
    comm_ctrl: int,
    comm_tgt: int,
    data_tgt: int,
    c_start: int,
    c_end: int,
    include_bell: bool = True,
    target_instructions: list[InstructionType] | None = None,
    source_final_instructions: list[InstructionType] | None = None,
) -> None:
    target_instructions = target_instructions if target_instructions is not None else instructions
    source_final_instructions = (
        source_final_instructions if source_final_instructions is not None else instructions
    )
    if include_bell:
        instructions.append(
            GateInstruction(
                name="bell_pair_phi_plus",
                qubits=[comm_ctrl, comm_tgt],
                params=[],
                opaque=True,
            )
        )
    instructions.append(
        UInstruction(
            qubit=comm_ctrl,
            qubits=[comm_ctrl],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(
        UInstruction(
            qubit=data_ctrl,
            qubits=[data_ctrl],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_ctrl,
            qubits=[comm_ctrl],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        RzzInstruction(
            a=data_ctrl,
            b=comm_ctrl,
            qubits=[data_ctrl, comm_ctrl],
            theta=pi / 2,
            params=[pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_ctrl,
            qubits=[comm_ctrl],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(MeasureInstruction(qubit=comm_ctrl, cbit=c_start, qubits=[comm_ctrl]))
    instructions.append(ResetInstruction(qubit=comm_ctrl, qubits=[comm_ctrl]))
    target_instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_start, value=True),
            op=UInstruction(
                qubit=comm_tgt,
                qubits=[comm_tgt],
                theta=pi,
                phi=0,
                lam=pi,
                params=[pi, 0, pi],
            ),
            qubits=[comm_tgt],
        )
    )
    target_instructions.append(
        UInstruction(
            qubit=comm_tgt,
            qubits=[comm_tgt],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    target_instructions.append(
        UInstruction(
            qubit=data_tgt,
            qubits=[data_tgt],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    target_instructions.append(
        RzzInstruction(
            a=comm_tgt,
            b=data_tgt,
            qubits=[comm_tgt, data_tgt],
            theta=pi / 2,
            params=[pi / 2],
        )
    )
    target_instructions.append(
        UInstruction(
            qubit=comm_tgt,
            qubits=[comm_tgt],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    target_instructions.append(MeasureInstruction(qubit=comm_tgt, cbit=c_end, qubits=[comm_tgt]))
    target_instructions.append(ResetInstruction(qubit=comm_tgt, qubits=[comm_tgt]))
    source_final_instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_end, value=True),
            op=UInstruction(
                qubit=data_ctrl,
                qubits=[data_ctrl],
                theta=0,
                phi=0,
                lam=pi,
                params=[0, 0, pi],
            ),
            qubits=[data_ctrl],
        )
    )


def _emit_teledata_protocol(
    instructions: list[InstructionType],
    *,
    data_src: int,
    comm_src: int,
    comm_dst: int,
    c_data: int,
    c_comm: int,
    include_bell: bool = True,
    dest_instructions: list[InstructionType] | None = None,
) -> None:
    dest_instructions = dest_instructions if dest_instructions is not None else instructions
    if include_bell:
        instructions.append(
            GateInstruction(
                name="bell_pair_phi_plus",
                qubits=[comm_src, comm_dst],
                params=[],
                opaque=True,
            )
        )
    instructions.append(
        UInstruction(
            qubit=comm_src,
            qubits=[comm_src],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(
        UInstruction(
            qubit=data_src,
            qubits=[data_src],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_src,
            qubits=[comm_src],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        RzzInstruction(
            a=data_src,
            b=comm_src,
            qubits=[data_src, comm_src],
            theta=pi / 2,
            params=[pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_src,
            qubits=[comm_src],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(
        UInstruction(
            qubit=data_src,
            qubits=[data_src],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(MeasureInstruction(qubit=data_src, cbit=c_data, qubits=[data_src]))
    instructions.append(ResetInstruction(qubit=data_src, qubits=[data_src]))
    instructions.append(MeasureInstruction(qubit=comm_src, cbit=c_comm, qubits=[comm_src]))
    instructions.append(ResetInstruction(qubit=comm_src, qubits=[comm_src]))
    dest_instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_comm, value=True),
            op=UInstruction(
                qubit=comm_dst,
                qubits=[comm_dst],
                theta=pi,
                phi=0,
                lam=pi,
                params=[pi, 0, pi],
            ),
            qubits=[comm_dst],
        )
    )
    dest_instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_data, value=True),
            op=UInstruction(
                qubit=comm_dst,
                qubits=[comm_dst],
                theta=0,
                phi=0,
                lam=pi,
                params=[0, 0, pi],
            ),
            qubits=[comm_dst],
        )
    )
