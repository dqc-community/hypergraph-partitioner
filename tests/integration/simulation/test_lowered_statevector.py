from __future__ import annotations

import pytest
from bosonic_converters import CircuitConverters
from bosonic_model import Circuit, ConditionalInstruction
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from hypergraph_partitioner import (
    build_annotated_circuit,
    count_nonlocal_interactions,
    count_teleports,
    lower_distributed_circuit,
    partition_circuit,
)
from hypergraph_partitioner.config import KAHYPAR_CONFIG
from hypergraph_partitioner.models.circuit_annotations import PartitionedCircuit
from tests.integration.simulation.statevector_test_utils import (
    INPUT_STATES,
    assert_statevectors_equivalent,
    embedded_original_to_qiskit,
    local_only_circuit,
    remote_cz_circuit,
    simulate_statevector,
    small_multi_segment_regression_circuit,
    teleport_regression_circuit,
    to_aer_compatible_qiskit,
    with_preparations,
)


def _simulate_statevector(circuit: QuantumCircuit) -> Statevector:
    circuit = to_aer_compatible_qiskit(circuit.copy())
    return simulate_statevector(circuit)


def _lowered_monolithic(partitioned: PartitionedCircuit, qpu_data_capacity: int) -> Circuit:
    symbolic = build_annotated_circuit(partitioned, qpu_data_capacity=qpu_data_capacity)
    lowered = lower_distributed_circuit(symbolic)
    return lowered.as_monolithic_circuit()


def _lowered_to_qiskit(partitioned: PartitionedCircuit, qpu_data_capacity: int) -> QuantumCircuit:
    return CircuitConverters.to_qiskit(_lowered_monolithic(partitioned, qpu_data_capacity))


def _instruction_names(circuit: Circuit) -> list[str]:
    return [str(getattr(inst, "name", getattr(inst, "kind", ""))) for inst in circuit.instructions]


def _contains_conditional(circuit: Circuit) -> bool:
    return any(isinstance(inst, ConditionalInstruction) for inst in circuit.instructions)


@pytest.mark.integration
@pytest.mark.parametrize("input_control", INPUT_STATES)
@pytest.mark.parametrize("input_target", INPUT_STATES)
def test_lowered_statevector_matches_original_for_local_only(
    input_control: str, input_target: str
) -> None:
    circuit = local_only_circuit(input_control, input_target)
    partitioned = partition_circuit(
        circuit,
        k=1,
        init_seg_size=10,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    lowered_monolithic = _lowered_monolithic(partitioned, qpu_data_capacity=2)
    names = _instruction_names(lowered_monolithic)
    assert "remote_bell_pair_phi_plus" not in names
    assert "measure" not in names
    assert "reset" not in names

    lowered = _simulate_statevector(_lowered_to_qiskit(partitioned, qpu_data_capacity=2))
    original = _simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qpu_data_capacity=2),
    )

    assert_statevectors_equivalent(lowered, original)


@pytest.mark.integration
@pytest.mark.parametrize("input_control", INPUT_STATES)
@pytest.mark.parametrize("input_target", INPUT_STATES)
def test_lowered_statevector_matches_original_for_remote_cz(
    input_control: str, input_target: str
) -> None:
    circuit = remote_cz_circuit(input_control, input_target)
    partitioned = partition_circuit(
        circuit,
        k=2,
        init_seg_size=10,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert count_nonlocal_interactions(partitioned) == 1
    assert count_teleports(partitioned) == 0

    lowered_monolithic = _lowered_monolithic(partitioned, qpu_data_capacity=1)
    names = _instruction_names(lowered_monolithic)
    assert "remote_bell_pair_phi_plus" in names
    assert "measure" in names
    assert "reset" in names

    lowered = _simulate_statevector(_lowered_to_qiskit(partitioned, qpu_data_capacity=1))
    original = _simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qpu_data_capacity=1),
    )

    assert_statevectors_equivalent(lowered, original)


@pytest.mark.integration
@pytest.mark.parametrize(
    "preparations",
    [
        [],
        [(0, "+")],
        [(1, "1"), (3, "+i")],
        [(0, "+"), (1, "+i"), (2, "1"), (3, "+")],
    ],
)
def test_lowered_statevector_matches_original_for_teleporting_circuit(
    preparations: list[tuple[int, str]]
) -> None:
    circuit = with_preparations(teleport_regression_circuit(), preparations)
    partitioned = partition_circuit(
        circuit,
        k=2,
        init_seg_size=2,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert len(partitioned.segments) == 2
    assert count_nonlocal_interactions(partitioned) == 2
    assert count_teleports(partitioned) == 2

    lowered_monolithic = _lowered_monolithic(partitioned, qpu_data_capacity=2)
    names = _instruction_names(lowered_monolithic)
    assert "remote_bell_pair_phi_plus" in names
    assert "measure" in names
    assert "reset" in names
    assert _contains_conditional(lowered_monolithic)

    lowered = _simulate_statevector(_lowered_to_qiskit(partitioned, qpu_data_capacity=2))
    original = _simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qpu_data_capacity=2),
    )

    assert_statevectors_equivalent(lowered, original)


@pytest.mark.integration
def test_lowered_statevector_matches_original_for_small_multi_segment_regression_circuit() -> None:
    circuit = small_multi_segment_regression_circuit()
    partitioned = partition_circuit(
        circuit,
        k=2,
        init_seg_size=2,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert len(partitioned.segments) == 2
    assert count_nonlocal_interactions(partitioned) == 3
    assert count_teleports(partitioned) == 2

    lowered_monolithic = _lowered_monolithic(partitioned, qpu_data_capacity=2)
    names = _instruction_names(lowered_monolithic)
    assert "remote_bell_pair_phi_plus" in names
    assert "measure" in names
    assert "reset" in names
    assert _contains_conditional(lowered_monolithic)

    lowered_circ = _lowered_to_qiskit(partitioned, qpu_data_capacity=2)

    lowered = _simulate_statevector(lowered_circ)
    original = _simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qpu_data_capacity=2),
    )

    assert_statevectors_equivalent(lowered, original)
