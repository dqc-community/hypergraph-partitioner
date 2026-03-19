"""Qiskit-native verification for telegate and teledata lowering protocols."""

from __future__ import annotations

from math import pi
import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity


_INPUT_STATES = ("0", "1", "+", "+i")


def _bell_pair_gate_phi_plus() -> UnitaryGate:
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


def _build_remote_cz_protocol_qiskit(input_control: str, input_target: str) -> QuantumCircuit:
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

    qc.append(_bell_pair_gate_phi_plus(), [comm_ctrl, comm_tgt])
    qc.h(comm_ctrl)
    qc.cz(data_ctrl, comm_ctrl)
    qc.h(comm_ctrl)
    qc.measure(comm_ctrl, c_start[0])
    qc.reset(comm_ctrl)
    with qc.if_test((c_start[0], 1)):
        qc.u(pi, 0, pi, comm_tgt)

    qc.u(0, 0, -pi / 2, comm_tgt)
    qc.u(0, 0, -pi / 2, data_tgt)
    qc.rzz(pi / 2, comm_tgt, data_tgt)

    qc.h(comm_tgt)
    qc.measure(comm_tgt, c_end[0])
    qc.reset(comm_tgt)
    with qc.if_test((c_end[0], 1)):
        qc.u(0, 0, pi, data_ctrl)

    qc.save_density_matrix()
    return qc


def _build_remote_cz_protocol_bosonic_basis_qiskit(
    input_control: str, input_target: str
) -> QuantumCircuit:
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

    qc.append(_bell_pair_gate_phi_plus(), [comm_ctrl, comm_tgt])
    qc.u(pi / 2, 0, pi, comm_ctrl)
    qc.u(0, 0, -pi / 2, data_ctrl)
    qc.u(0, 0, -pi / 2, comm_ctrl)
    qc.rzz(pi / 2, data_ctrl, comm_ctrl)
    qc.u(pi / 2, 0, pi, comm_ctrl)
    qc.measure(comm_ctrl, c_start[0])
    qc.reset(comm_ctrl)
    with qc.if_test((c_start[0], 1)):
        qc.u(pi, 0, pi, comm_tgt)

    qc.u(0, 0, -pi / 2, comm_tgt)
    qc.u(0, 0, -pi / 2, data_tgt)
    qc.rzz(pi / 2, comm_tgt, data_tgt)

    qc.u(pi / 2, 0, pi, comm_tgt)
    qc.measure(comm_tgt, c_end[0])
    qc.reset(comm_tgt)
    with qc.if_test((c_end[0], 1)):
        qc.u(0, 0, pi, data_ctrl)

    qc.save_density_matrix()
    return qc


def _build_ideal_cz(input_control: str, input_target: str) -> QuantumCircuit:
    qc = QuantumCircuit(4)
    _prepare_state(qc, 0, input_control)
    _prepare_state(qc, 3, input_target)
    qc.cz(0, 3)
    qc.save_density_matrix()
    return qc


def _build_teledata_protocol_qiskit(input_state: str) -> QuantumCircuit:
    q = QuantumRegister(3, "q")
    c_data = ClassicalRegister(1, "c_data")
    c_comm = ClassicalRegister(1, "c_comm")
    qc = QuantumCircuit(q, c_data, c_comm)

    data_src = q[0]
    comm_src = q[1]
    comm_dst = q[2]

    _prepare_state(qc, data_src, input_state)

    qc.append(_bell_pair_gate_phi_plus(), [comm_src, comm_dst])
    qc.u(pi / 2, 0, pi, comm_src)
    qc.u(0, 0, -pi / 2, data_src)
    qc.u(0, 0, -pi / 2, comm_src)
    qc.rzz(pi / 2, data_src, comm_src)
    qc.u(pi / 2, 0, pi, comm_src)
    qc.u(pi / 2, 0, pi, data_src)
    qc.measure(data_src, c_data[0])
    qc.reset(data_src)
    qc.measure(comm_src, c_comm[0])
    qc.reset(comm_src)
    with qc.if_test((c_comm[0], 1)):
        qc.u(pi, 0, pi, comm_dst)
    with qc.if_test((c_data[0], 1)):
        qc.u(0, 0, pi, comm_dst)

    qc.save_density_matrix()
    return qc


def _build_ideal_teledata(input_state: str) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    _prepare_state(qc, 2, input_state)
    qc.save_density_matrix()
    return qc


@pytest.mark.parametrize("input_control", _INPUT_STATES)
@pytest.mark.parametrize("input_target", _INPUT_STATES)
def test_cat_entangler_remote_cz_matches_ideal_cz_qiskit(
    input_control: str, input_target: str
) -> None:
    simulator = AerSimulator(method="density_matrix")

    remote_result = simulator.run(_build_remote_cz_protocol_qiskit(input_control, input_target)).result()
    ideal_result = simulator.run(_build_ideal_cz(input_control, input_target)).result()

    remote_density = DensityMatrix(remote_result.data(0)["density_matrix"])
    ideal_density = DensityMatrix(ideal_result.data(0)["density_matrix"])

    reduced_remote = partial_trace(remote_density, [1, 2])
    reduced_ideal = partial_trace(ideal_density, [1, 2])

    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("input_control", _INPUT_STATES)
@pytest.mark.parametrize("input_target", _INPUT_STATES)
def test_cat_entangler_remote_cz_in_bosonic_basis_matches_ideal_cz_qiskit(
    input_control: str, input_target: str
) -> None:
    simulator = AerSimulator(method="density_matrix")

    remote_result = simulator.run(
        _build_remote_cz_protocol_bosonic_basis_qiskit(input_control, input_target)
    ).result()
    ideal_result = simulator.run(_build_ideal_cz(input_control, input_target)).result()

    remote_density = DensityMatrix(remote_result.data(0)["density_matrix"])
    ideal_density = DensityMatrix(ideal_result.data(0)["density_matrix"])

    reduced_remote = partial_trace(remote_density, [1, 2])
    reduced_ideal = partial_trace(ideal_density, [1, 2])

    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("input_state", _INPUT_STATES)
def test_teledata_state_transfer_matches_ideal_destination_state_qiskit(input_state: str) -> None:
    simulator = AerSimulator(method="density_matrix")

    remote_result = simulator.run(_build_teledata_protocol_qiskit(input_state)).result()
    ideal_result = simulator.run(_build_ideal_teledata(input_state)).result()

    remote_density = DensityMatrix(remote_result.data(0)["density_matrix"])
    ideal_density = DensityMatrix(ideal_result.data(0)["density_matrix"])

    reduced_remote = partial_trace(remote_density, [0, 1])
    reduced_ideal = partial_trace(ideal_density, [0, 1])

    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)
