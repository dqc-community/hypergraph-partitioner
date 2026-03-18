"""Build symbolic distributed circuits from annotated partitioned circuits."""

from __future__ import annotations

from bosonic_model import (
    Circuit,
    ConditionalInstruction,
    DistributedCircuit,
    GateInstruction,
)

from hypergraph_partitioner.bosonic_pipeline import iter_annotated_operations
from hypergraph_partitioner.models.circuit_annotations import (
    BoundaryTeleportOp,
    LocalOp,
    NonlocalCZOp,
    PartitionedCircuit,
)
from hypergraph_partitioner.qpu_utils import (
    alloc_receiver,
    append_instruction,
    append_shared_instruction,
    build_qpu_layouts,
    finalize_circuit_registers,
    max_existing_cbit,
    num_nodes,
    remap_instruction,
    validate_capacity,
)

from .state import DistributionState, PhysicalLocation


def build_annotated_circuit(
    partitioned: PartitionedCircuit, *, qpu_data_capacity: int
) -> DistributedCircuit:
    if qpu_data_capacity < 1:
        raise ValueError("qpu_data_capacity must be positive")
    if not partitioned.segments:
        return DistributedCircuit(qubits_per_node={}, circuits={})

    n_nodes = num_nodes(partitioned)
    validate_capacity(partitioned, qpu_data_capacity)
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

    finalize_circuit_registers(
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


def _initialize_distribution_state(
    partitioned: PartitionedCircuit, qpu_data_capacity: int, n_nodes: int
) -> DistributionState:
    qpu_layouts = build_qpu_layouts(qpu_data_capacity, n_nodes)
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
        next_cbit=max_existing_cbit(partitioned) + 1,
    )


def _distribute_local(op: LocalOp, state: DistributionState) -> None:
    inst = op.instruction
    inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
    qubits = list(getattr(inner, "qubits", []) or [])
    qubit_map = {
        qubit: state.qubit_locations[qubit].qubit
        for qubit in qubits
        if qubit in state.qubit_locations
    }
    mapped = remap_instruction(inst, qubit_map)
    node = state.qubit_locations[qubits[0]].node if qubits else 0
    append_instruction(state.circuits, state.instruction_index, node, mapped, state)


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
    append_shared_instruction(
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
    dst_qubit = alloc_receiver(dst_layout)
    inst = GateInstruction(name="teleport", qubits=[source.qubit, dst_qubit], params=[], opaque=True)
    append_shared_instruction(
        state.circuits,
        state.instruction_index,
        (source.node, int(op.to_node)),
        inst,
        state,
    )
    state.qubit_locations[qubit] = PhysicalLocation(node=int(op.to_node), qubit=dst_qubit)
