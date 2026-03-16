"""DSL scaffolding for telegate and teledata lowering protocols."""

from __future__ import annotations

from math import pi

from bosonic_converters import CircuitConverters
from bosonic_model import (
    Circuit,
    Condition,
    ConditionalInstruction,
    GateInstruction,
    MeasureInstruction,
    Register,
    ResetInstruction,
    RzzInstruction,
    UInstruction,
)
import pytest
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity


def _registers(*, n_qubits: int, classical: tuple[tuple[str, int], ...]) -> tuple[dict[str, Register], dict[str, Register]]:
    qregs = {"q": Register(name="q", size=n_qubits, base=0)}
    cregs: dict[str, Register] = {}
    base = 0
    for name, size in classical:
        cregs[name] = Register(name=name, size=size, base=base)
        base += size
    return qregs, cregs


def _prepare_state_dsl(instructions: list, qubit: int, label: str) -> None:
    if label == "0":
        return
    if label == "1":
        instructions.append(UInstruction(qubit=qubit, qubits=[qubit], theta=pi, phi=0, lam=pi, params=[pi, 0, pi]))
        return
    if label == "+":
        instructions.append(UInstruction(qubit=qubit, qubits=[qubit], theta=pi / 2, phi=0, lam=pi, params=[pi / 2, 0, pi]))
        return
    if label == "+i":
        instructions.append(UInstruction(qubit=qubit, qubits=[qubit], theta=pi / 2, phi=0, lam=pi, params=[pi / 2, 0, pi]))
        instructions.append(UInstruction(qubit=qubit, qubits=[qubit], theta=0, phi=0, lam=pi / 2, params=[0, 0, pi / 2]))
        return
    raise AssertionError(f"unexpected state label: {label}")


_INPUT_STATES = ("0", "1", "+", "+i")


def _build_remote_cz_protocol_dsl(input_control: str, input_target: str) -> Circuit:
    qregs, cregs = _registers(n_qubits=4, classical=(("c_start", 1), ("c_end", 1)))
    instructions: list = []

    data_ctrl = 0
    comm_ctrl = 1
    comm_tgt = 2
    data_tgt = 3
    c_start = 0
    c_end = 1

    _prepare_state_dsl(instructions, data_ctrl, input_control)
    _prepare_state_dsl(instructions, data_tgt, input_target)

    instructions.append(
        GateInstruction(
            name="remote_link_psi_plus",
            qubits=[comm_ctrl, comm_tgt],
            params=[],
            opaque=True,
        )
    )
    instructions.append(UInstruction(qubit=comm_ctrl, qubits=[comm_ctrl], theta=pi / 2, phi=0, lam=pi, params=[pi / 2, 0, pi]))
    instructions.append(UInstruction(qubit=data_ctrl, qubits=[data_ctrl], theta=0, phi=0, lam=-pi / 2, params=[0, 0, -pi / 2]))
    instructions.append(UInstruction(qubit=comm_ctrl, qubits=[comm_ctrl], theta=0, phi=0, lam=-pi / 2, params=[0, 0, -pi / 2]))
    instructions.append(RzzInstruction(a=data_ctrl, b=comm_ctrl, qubits=[data_ctrl, comm_ctrl], theta=pi / 2, params=[pi / 2]))
    instructions.append(UInstruction(qubit=comm_ctrl, qubits=[comm_ctrl], theta=pi / 2, phi=0, lam=pi, params=[pi / 2, 0, pi]))
    instructions.append(MeasureInstruction(qubit=comm_ctrl, cbit=c_start, qubits=[comm_ctrl]))
    instructions.append(ResetInstruction(qubit=comm_ctrl, qubits=[comm_ctrl]))
    instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_start, value=True),
            op=UInstruction(qubit=comm_tgt, qubits=[comm_tgt], theta=pi, phi=0, lam=pi, params=[pi, 0, pi]),
            qubits=[comm_tgt],
        )
    )
    instructions.append(UInstruction(qubit=comm_tgt, qubits=[comm_tgt], theta=0, phi=0, lam=-pi / 2, params=[0, 0, -pi / 2]))
    instructions.append(UInstruction(qubit=data_tgt, qubits=[data_tgt], theta=0, phi=0, lam=-pi / 2, params=[0, 0, -pi / 2]))
    instructions.append(RzzInstruction(a=comm_tgt, b=data_tgt, qubits=[comm_tgt, data_tgt], theta=pi / 2, params=[pi / 2]))
    instructions.append(UInstruction(qubit=comm_tgt, qubits=[comm_tgt], theta=pi / 2, phi=0, lam=pi, params=[pi / 2, 0, pi]))
    instructions.append(MeasureInstruction(qubit=comm_tgt, cbit=c_end, qubits=[comm_tgt]))
    instructions.append(ResetInstruction(qubit=comm_tgt, qubits=[comm_tgt]))
    instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_end, value=True),
            op=UInstruction(qubit=data_ctrl, qubits=[data_ctrl], theta=0, phi=0, lam=pi, params=[0, 0, pi]),
            qubits=[data_ctrl],
        )
    )

    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def _build_teledata_protocol_dsl(input_state: str) -> Circuit:
    qregs, cregs = _registers(n_qubits=3, classical=(("c_data", 1), ("c_comm", 1)))
    instructions: list = []

    data_src = 0
    comm_src = 1
    comm_dst = 2
    c_data = 0
    c_comm = 1

    _prepare_state_dsl(instructions, data_src, input_state)
    instructions.append(
        GateInstruction(
            name="remote_link_psi_plus",
            qubits=[comm_src, comm_dst],
            params=[],
            opaque=True,
        )
    )
    instructions.append(UInstruction(qubit=comm_src, qubits=[comm_src], theta=pi / 2, phi=0, lam=pi, params=[pi / 2, 0, pi]))
    instructions.append(UInstruction(qubit=data_src, qubits=[data_src], theta=0, phi=0, lam=-pi / 2, params=[0, 0, -pi / 2]))
    instructions.append(UInstruction(qubit=comm_src, qubits=[comm_src], theta=0, phi=0, lam=-pi / 2, params=[0, 0, -pi / 2]))
    instructions.append(RzzInstruction(a=data_src, b=comm_src, qubits=[data_src, comm_src], theta=pi / 2, params=[pi / 2]))
    instructions.append(UInstruction(qubit=comm_src, qubits=[comm_src], theta=pi / 2, phi=0, lam=pi, params=[pi / 2, 0, pi]))
    instructions.append(UInstruction(qubit=data_src, qubits=[data_src], theta=pi / 2, phi=0, lam=pi, params=[pi / 2, 0, pi]))
    instructions.append(MeasureInstruction(qubit=data_src, cbit=c_data, qubits=[data_src]))
    instructions.append(ResetInstruction(qubit=data_src, qubits=[data_src]))
    instructions.append(MeasureInstruction(qubit=comm_src, cbit=c_comm, qubits=[comm_src]))
    instructions.append(ResetInstruction(qubit=comm_src, qubits=[comm_src]))
    instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_comm, value=True),
            op=UInstruction(qubit=comm_dst, qubits=[comm_dst], theta=pi, phi=0, lam=pi, params=[pi, 0, pi]),
            qubits=[comm_dst],
        )
    )
    instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_data, value=True),
            op=UInstruction(qubit=comm_dst, qubits=[comm_dst], theta=0, phi=0, lam=pi, params=[0, 0, pi]),
            qubits=[comm_dst],
        )
    )
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def _build_ideal_teledata_dsl(input_state: str) -> Circuit:
    qregs, cregs = _registers(n_qubits=3, classical=())
    instructions: list = []
    _prepare_state_dsl(instructions, 2, input_state)
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def test_remote_cz_protocol_dsl_converts_to_qiskit() -> None:
    circuit = _build_remote_cz_protocol_dsl("0", "0")
    qiskit_circuit = CircuitConverters.to_qiskit(circuit)
    assert qiskit_circuit.num_qubits == 4
    assert qiskit_circuit.num_clbits == 2


def test_teledata_protocol_dsl_converts_to_qiskit() -> None:
    circuit = _build_teledata_protocol_dsl("0")
    qiskit_circuit = CircuitConverters.to_qiskit(circuit)
    assert qiskit_circuit.num_qubits == 3
    assert qiskit_circuit.num_clbits == 2


@pytest.mark.parametrize("input_state", _INPUT_STATES)
def test_teledata_state_transfer_matches_ideal_destination_state_via_dsl(input_state: str) -> None:
    simulator = AerSimulator(method="density_matrix")

    remote_qiskit = CircuitConverters.to_qiskit(_build_teledata_protocol_dsl(input_state))
    ideal_qiskit = CircuitConverters.to_qiskit(_build_ideal_teledata_dsl(input_state))

    remote_qiskit.save_density_matrix()
    ideal_qiskit.save_density_matrix()

    remote_result = simulator.run(remote_qiskit).result()
    ideal_result = simulator.run(ideal_qiskit).result()

    remote_density = DensityMatrix(remote_result.data(0)["density_matrix"])
    ideal_density = DensityMatrix(ideal_result.data(0)["density_matrix"])

    reduced_remote = partial_trace(remote_density, [0, 1])
    reduced_ideal = partial_trace(ideal_density, [0, 1])

    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)
