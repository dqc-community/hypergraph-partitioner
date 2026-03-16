"""Aer-based verification for a cat-entangler/disentangler remote CZ protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity


_INPUT_STATES = ("0", "1", "+", "+i")


def _bell_pair_gate_phi_plus() -> UnitaryGate:
    """Primitive Bell-pair preparation gate, represented as a named unitary."""

    matrix = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, -1],
            [1, 0, -1, 0],
        ],
        dtype=complex,
    ) * (1 / np.sqrt(2))
    return UnitaryGate(matrix, label="bell_pair_phi_plus")


def _prepare_state(qc: QuantumCircuit, qubit, label: str) -> None:
    if label == "0":
        return
    if label == "1":
        qc.x(qubit)
        return
    if label == "+":
        qc.h(qubit)
        return
    if label == "+i":
        qc.h(qubit)
        qc.s(qubit)
        return
    raise AssertionError(f"unexpected state label: {label}")


def _build_remote_cz_protocol(input_control: str, input_target: str) -> QuantumCircuit:
    q = QuantumRegister(4, "q")
    c_start = ClassicalRegister(1, "c_start")
    c_end = ClassicalRegister(1, "c_end")
    qc = QuantumCircuit(q, c_start, c_end)

    data_ctrl = q[0]
    comm_ctrl = q[1]
    comm_tgt = q[2]
    data_tgt = q[3]

    _prepare_state(qc, data_ctrl, input_control)
    _prepare_state(qc, data_tgt, input_target)

    # Starting process: cat entangler / non-local fan-out.
    qc.append(_bell_pair_gate_phi_plus(), [comm_ctrl, comm_tgt])
    qc.cx(data_ctrl, comm_ctrl)
    qc.measure(comm_ctrl, c_start[0])
    qc.reset(comm_ctrl)
    with qc.if_test((c_start[0], 1)):
        qc.x(comm_tgt)

    # Execute the requested gate locally at the target side.
    qc.cz(comm_tgt, data_tgt)

    # Ending process: cat disentangler / non-local fan-in.
    qc.h(comm_tgt)
    qc.measure(comm_tgt, c_end[0])
    qc.reset(comm_tgt)
    with qc.if_test((c_end[0], 1)):
        qc.z(data_ctrl)

    qc.save_density_matrix()
    return qc


def _build_ideal_cz(input_control: str, input_target: str) -> QuantumCircuit:
    qc = QuantumCircuit(4)
    _prepare_state(qc, 0, input_control)
    _prepare_state(qc, 3, input_target)
    qc.cz(0, 3)
    qc.save_density_matrix()
    return qc


def test_remote_cz_protocol_diagram_is_renderable_with_mpl() -> None:
    artifact_dir = Path(".pytest_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    figure = _build_remote_cz_protocol("0", "0").draw(output="mpl")
    output_path = artifact_dir / "remote_cz_protocol.png"
    figure.savefig(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    figure.clf()


@pytest.mark.parametrize("input_control", _INPUT_STATES)
@pytest.mark.parametrize("input_target", _INPUT_STATES)
def test_cat_entangler_remote_cz_matches_ideal_cz(
    input_control: str, input_target: str
) -> None:
    simulator = AerSimulator(method="density_matrix")

    remote_result = simulator.run(_build_remote_cz_protocol(input_control, input_target)).result()
    ideal_result = simulator.run(_build_ideal_cz(input_control, input_target)).result()

    remote_density = DensityMatrix(remote_result.data(0)["density_matrix"])
    ideal_density = DensityMatrix(ideal_result.data(0)["density_matrix"])

    # Communication qubits are ancillas; compare only the logical data qubits.
    reduced_remote = partial_trace(remote_density, [1, 2])
    reduced_ideal = partial_trace(ideal_density, [1, 2])

    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)
