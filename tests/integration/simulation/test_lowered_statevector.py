from __future__ import annotations

import pytest
from bosonic_converters import CircuitConverters
from bosonic_model import Circuit, ConditionalInstruction
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from hypergraph_partitioner import (
    build_annotated_circuit,
    lower_distributed_circuit,
)
from hypergraph_partitioner.bosonic_pipeline import (
    _count_nonlocal_interactions,
    _count_teleports,
    _partition_to_partitioned_circuit,
)
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


def _lowered_monolithic(partitioned: PartitionedCircuit, qubits_per_node: int) -> Circuit:
    symbolic = build_annotated_circuit(partitioned, qubits_per_node=qubits_per_node)
    lowered = lower_distributed_circuit(symbolic)
    return lowered.as_monolithic_circuit()


def _lowered_to_qiskit(partitioned: PartitionedCircuit, qubits_per_node: int) -> QuantumCircuit:
    return CircuitConverters.to_qiskit(_lowered_monolithic(partitioned, qubits_per_node))


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
    partitioned = _partition_to_partitioned_circuit(
        circuit,
        nodes=1,
        init_seg_size=10,
        max_hedge_dist=100,
    )

    lowered_monolithic = _lowered_monolithic(partitioned, qubits_per_node=2)
    names = _instruction_names(lowered_monolithic)
    assert "remote_link_phi_plus" not in names
    assert "measure" not in names
    assert "reset" not in names

    lowered = _simulate_statevector(_lowered_to_qiskit(partitioned, qubits_per_node=2))
    original = _simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qubits_per_node=2),
    )

    assert_statevectors_equivalent(lowered, original)


@pytest.mark.integration
@pytest.mark.parametrize("input_control", INPUT_STATES)
@pytest.mark.parametrize("input_target", INPUT_STATES)
def test_lowered_statevector_matches_original_for_remote_cz(
    input_control: str, input_target: str
) -> None:
    circuit = remote_cz_circuit(input_control, input_target)
    partitioned = _partition_to_partitioned_circuit(
        circuit,
        nodes=2,
        init_seg_size=10,
        max_hedge_dist=100,
    )

    assert _count_nonlocal_interactions(partitioned) == 1
    assert _count_teleports(partitioned) == 0

    lowered_monolithic = _lowered_monolithic(partitioned, qubits_per_node=1)
    names = _instruction_names(lowered_monolithic)
    assert "remote_link_phi_plus" in names
    assert "measure" in names
    assert "reset" in names

    lowered = _simulate_statevector(_lowered_to_qiskit(partitioned, qubits_per_node=1))
    original = _simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qubits_per_node=1),
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
    partitioned = _partition_to_partitioned_circuit(
        circuit,
        nodes=2,
        init_seg_size=2,
        max_hedge_dist=100,
    )

    assert len(partitioned.segments) == 2
    assert _count_nonlocal_interactions(partitioned) == 2
    assert _count_teleports(partitioned) == 2

    lowered_monolithic = _lowered_monolithic(partitioned, qubits_per_node=2)
    names = _instruction_names(lowered_monolithic)
    assert "remote_link_phi_plus" in names
    assert "measure" in names
    assert "reset" in names
    assert _contains_conditional(lowered_monolithic)

    lowered = _simulate_statevector(_lowered_to_qiskit(partitioned, qubits_per_node=2))
    original = _simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qubits_per_node=2),
    )

    assert_statevectors_equivalent(lowered, original)


@pytest.mark.integration
def test_lowered_statevector_matches_original_for_small_multi_segment_regression_circuit() -> None:
    circuit = small_multi_segment_regression_circuit()
    partitioned = _partition_to_partitioned_circuit(
        circuit,
        nodes=2,
        init_seg_size=2,
        max_hedge_dist=100,
    )

    assert len(partitioned.segments) == 2
    assert _count_nonlocal_interactions(partitioned) == 3
    assert _count_teleports(partitioned) == 2

    lowered_monolithic = _lowered_monolithic(partitioned, qubits_per_node=2)
    names = _instruction_names(lowered_monolithic)
    assert "remote_link_phi_plus" in names
    assert "measure" in names
    assert "reset" in names
    assert _contains_conditional(lowered_monolithic)

    lowered_circ = _lowered_to_qiskit(partitioned, qubits_per_node=2)

    lowered = _simulate_statevector(lowered_circ)
    original = _simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qubits_per_node=2),
    )

    assert_statevectors_equivalent(lowered, original)
