from __future__ import annotations

from math import pi

from bosonic_converters import CircuitConverters
from bosonic_model import Circuit, Register, UInstruction
from bosonic_model.instructions import CzInstruction, InstructionType
from bosonic_model.qasm import Translator
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from hypergraph_partitioner.models.annotated import BoundaryTeleportOp, PartitionedCircuit


INPUT_STATES = ("0", "1", "+", "+i")


def u(qubit: int, theta: float, phi: float, lam: float) -> UInstruction:
    return UInstruction(
        qubit=qubit,
        qubits=[qubit],
        theta=theta,
        phi=phi,
        lam=lam,
        params=[theta, phi, lam],
    )


def prepare_label_ops(qubit: int, label: str) -> list[UInstruction]:
    if label == "0":
        return []
    if label == "1":
        return [u(qubit, pi, 0, pi)]
    if label == "+":
        return [u(qubit, pi / 2, 0, pi)]
    if label == "+i":
        return [u(qubit, pi / 2, 0, pi), u(qubit, 0, 0, pi / 2)]
    raise AssertionError(f"unexpected label {label}")


def with_preparations(circuit: Circuit, preparations: list[tuple[int, str]]) -> Circuit:
    instructions: list[InstructionType] = []
    for qubit, label in preparations:
        instructions.extend(prepare_label_ops(qubit, label))
    instructions.extend(circuit.instructions)
    return Circuit(qregs=circuit.qregs, cregs=circuit.cregs, instructions=instructions)


def simulate_statevector(circuit: QuantumCircuit) -> Statevector:
    simulator = AerSimulator(method="statevector")
    circuit = circuit.copy()
    circuit.save_statevector()
    result = simulator.run(circuit).result()
    return Statevector(result.data(0)["statevector"])


def assert_statevectors_equivalent(actual: Statevector, expected: Statevector) -> None:
    assert actual.equiv(expected)


def num_blocks(partitioned: PartitionedCircuit) -> int:
    return max(int(block) for seg in partitioned.segments for block in seg.partition.values()) + 1


def initial_wire_locations(
    partitioned: PartitionedCircuit, qpu_data_capacity: int
) -> dict[int, int]:
    n_blocks = num_blocks(partitioned)
    first_segment = partitioned.segments[0]
    locations: dict[int, int] = {}
    for block in range(n_blocks):
        wires = sorted(
            int(wire) for wire, owner in first_segment.partition.items() if int(owner) == block
        )
        base = block * 3 * qpu_data_capacity
        data_slots = list(range(base, base + qpu_data_capacity))
        for slot, wire in zip(data_slots, wires, strict=False):
            locations[wire] = slot
    return locations


def final_wire_locations(
    partitioned: PartitionedCircuit, qpu_data_capacity: int
) -> dict[int, int]:
    n_blocks = num_blocks(partitioned)
    locations = initial_wire_locations(partitioned, qpu_data_capacity)
    free_receivers = {
        block: set(
            range(
                block * 3 * qpu_data_capacity + 2 * qpu_data_capacity,
                (block + 1) * 3 * qpu_data_capacity,
            )
        )
        for block in range(n_blocks)
    }
    for op in partitioned.operations:
        if not isinstance(op, BoundaryTeleportOp):
            continue
        destination_block = int(op.to_block)
        destination = min(free_receivers[destination_block])
        free_receivers[destination_block].remove(destination)
        locations[int(op.wire)] = destination
    return locations


def embedded_original_to_qiskit(
    circuit: Circuit, partitioned: PartitionedCircuit, qpu_data_capacity: int
) -> QuantumCircuit:
    initial_locations_map = initial_wire_locations(partitioned, qpu_data_capacity)
    final_locations_map = final_wire_locations(partitioned, qpu_data_capacity)
    total_qubits = num_blocks(partitioned) * 3 * qpu_data_capacity

    logical = CircuitConverters.to_qiskit(circuit)
    embedded = QuantumCircuit(total_qubits)
    embedded.compose(
        logical,
        qubits=[initial_locations_map[wire] for wire in range(circuit.qubits())],
        inplace=True,
    )

    for wire in sorted(initial_locations_map):
        source = initial_locations_map[wire]
        destination = final_locations_map[wire]
        if source != destination:
            embedded.swap(source, destination)

    return embedded


def remote_cz_circuit(input_control: str, input_target: str) -> Circuit:
    qregs = {"q": Register(name="q", size=2, base=0)}
    instructions: list[InstructionType] = []
    instructions.extend(prepare_label_ops(0, input_control))
    instructions.extend(prepare_label_ops(1, input_target))
    instructions.append(CzInstruction(control=0, target=1, qubits=[0, 1]))
    return Circuit(qregs=qregs, cregs={}, instructions=instructions)


def local_only_circuit(input_control: str, input_target: str) -> Circuit:
    qregs = {"q": Register(name="q", size=2, base=0)}
    instructions: list[InstructionType] = []
    instructions.extend(prepare_label_ops(0, input_control))
    instructions.extend(prepare_label_ops(1, input_target))
    instructions.append(u(0, 0, 0, pi))
    instructions.append(u(1, pi / 2, 0, pi))
    return Circuit(qregs=qregs, cregs={}, instructions=instructions)


def teleport_regression_circuit() -> Circuit:
    return Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        y q[3];
        swap q[0],q[2];
        swap q[1],q[3];
        x q[3];
        swap q[3],q[0];
        cz q[3],q[0];
        swap q[0],q[1];
        x q[1];
        """
    )


def small_multi_segment_regression_circuit() -> Circuit:
    return Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        y q[1];
        s q[3];
        swap q[1],q[2];
        swap q[2],q[3];
        swap q[3],q[1];
        s q[0];
        s q[3];
        s q[3];
        h q[3];
        s q[2];
        """
    )


def multi_segment_regression_circuit() -> Circuit:
    return Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[8];
        cx q[6],q[0];
        s q[7];
        cx q[4],q[3];
        t q[3];
        swap q[2],q[7];
        y q[1];
        ccx q[4],q[7],q[5];
        ccx q[2],q[7],q[0];
        x q[5];
        cz q[1],q[2];
        cx q[5],q[4];
        z q[7];
        cz q[4],q[0];
        swap q[0],q[7];
        cx q[0],q[4];
        cz q[5],q[1];
        t q[1];
        z q[3];
        z q[2];
        swap q[7],q[0];
        x q[5];
        swap q[7],q[0];
        s q[4];
        x q[5];
        swap q[3],q[6];
        ccx q[4],q[3],q[0];
        ccx q[6],q[2],q[4];
        z q[4];
        y q[3];
        y q[0];
        ccx q[4],q[3],q[0];
        x q[2];
        y q[0];
        x q[6];
        swap q[4],q[7];
        z q[3];
        ccx q[6],q[4],q[2];
        cz q[7],q[5];
        t q[1];
        t q[1];
        cz q[5],q[6];
        z q[3];
        h q[4];
        x q[3];
        t q[2];
        t q[6];
        h q[1];
        y q[3];
        h q[1];
        h q[1];
        z q[1];
        cx q[1],q[2];
        x q[0];
        ccx q[0],q[1],q[6];
        x q[7];
        z q[0];
        h q[6];
        ccx q[1],q[6],q[2];
        x q[3];
        x q[4];
        """
    )
