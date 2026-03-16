"""DSL-based verification for telegate and teledata lowering protocols."""

from __future__ import annotations

from math import pi

import pytest
from bosonic_converters import CircuitConverters
from bosonic_model import Circuit, UInstruction
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity

from hypergraph_partitioner.lowering import (
    build_ideal_remote_cz_dsl,
    build_ideal_teledata_dsl,
    build_telegate_remote_cz_dsl,
    build_teledata_dsl,
    to_aer_compatible_qiskit,
)


_INPUT_STATES = ("0", "1", "+", "+i")


def _prepare_state_dsl(instructions: list, qubit: int, label: str) -> None:
    if label == "0":
        return
    if label == "1":
        instructions.append(
            UInstruction(
                qubit=qubit,
                qubits=[qubit],
                theta=pi,
                phi=0,
                lam=pi,
                params=[pi, 0, pi],
            )
        )
        return
    if label == "+":
        instructions.append(
            UInstruction(
                qubit=qubit,
                qubits=[qubit],
                theta=pi / 2,
                phi=0,
                lam=pi,
                params=[pi / 2, 0, pi],
            )
        )
        return
    if label == "+i":
        instructions.append(
            UInstruction(
                qubit=qubit,
                qubits=[qubit],
                theta=pi / 2,
                phi=0,
                lam=pi,
                params=[pi / 2, 0, pi],
            )
        )
        instructions.append(
            UInstruction(
                qubit=qubit,
                qubits=[qubit],
                theta=0,
                phi=0,
                lam=pi / 2,
                params=[0, 0, pi / 2],
            )
        )
        return
    raise AssertionError(f"unexpected state label: {label}")


def _with_prep(circuit: Circuit, preparations: list[tuple[int, str]]) -> Circuit:
    instructions: list = []
    for qubit, label in preparations:
        _prepare_state_dsl(instructions, qubit, label)
    instructions.extend(circuit.instructions)
    return Circuit(qregs=circuit.qregs, cregs=circuit.cregs, instructions=instructions)


def test_remote_cz_protocol_dsl_converts_to_qiskit() -> None:
    circuit = build_telegate_remote_cz_dsl()
    qiskit_circuit = CircuitConverters.to_qiskit(circuit)
    assert qiskit_circuit.num_qubits == 4
    assert qiskit_circuit.num_clbits == 2


@pytest.mark.parametrize("input_control", _INPUT_STATES)
@pytest.mark.parametrize("input_target", _INPUT_STATES)
def test_cat_entangler_remote_cz_matches_ideal_cz_via_dsl(
    input_control: str, input_target: str
) -> None:
    simulator = AerSimulator(method="density_matrix")

    remote_qiskit = to_aer_compatible_qiskit(
        CircuitConverters.to_qiskit(
            _with_prep(
                build_telegate_remote_cz_dsl(),
                [(0, input_control), (3, input_target)],
            )
        )
    )
    ideal_qiskit = to_aer_compatible_qiskit(
        CircuitConverters.to_qiskit(
            _with_prep(
                build_ideal_remote_cz_dsl(),
                [(0, input_control), (3, input_target)],
            )
        )
    )

    remote_qiskit.save_density_matrix()
    ideal_qiskit.save_density_matrix()

    remote_result = simulator.run(remote_qiskit).result()
    ideal_result = simulator.run(ideal_qiskit).result()

    remote_density = DensityMatrix(remote_result.data(0)["density_matrix"])
    ideal_density = DensityMatrix(ideal_result.data(0)["density_matrix"])

    reduced_remote = partial_trace(remote_density, [1, 2])
    reduced_ideal = partial_trace(ideal_density, [1, 2])

    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)


def test_teledata_protocol_dsl_converts_to_qiskit() -> None:
    circuit = build_teledata_dsl()
    qiskit_circuit = CircuitConverters.to_qiskit(circuit)
    assert qiskit_circuit.num_qubits == 3
    assert qiskit_circuit.num_clbits == 2


@pytest.mark.parametrize("input_state", _INPUT_STATES)
def test_teledata_state_transfer_matches_ideal_destination_state_via_dsl(
    input_state: str,
) -> None:
    simulator = AerSimulator(method="density_matrix")

    remote_qiskit = to_aer_compatible_qiskit(
        CircuitConverters.to_qiskit(_with_prep(build_teledata_dsl(), [(0, input_state)]))
    )
    ideal_qiskit = to_aer_compatible_qiskit(
        CircuitConverters.to_qiskit(_with_prep(build_ideal_teledata_dsl(), [(2, input_state)]))
    )

    remote_qiskit.save_density_matrix()
    ideal_qiskit.save_density_matrix()

    remote_result = simulator.run(remote_qiskit).result()
    ideal_result = simulator.run(ideal_qiskit).result()

    remote_density = DensityMatrix(remote_result.data(0)["density_matrix"])
    ideal_density = DensityMatrix(ideal_result.data(0)["density_matrix"])

    reduced_remote = partial_trace(remote_density, [0, 1])
    reduced_ideal = partial_trace(ideal_density, [0, 1])

    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)
