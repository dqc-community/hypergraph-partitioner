from __future__ import annotations

import pytest
from bosonic_converters import CircuitConverters
from bosonic_model import Circuit, GateInstruction
from qiskit import QuantumCircuit

from hypergraph_partitioner import (
    annotated_to_distributed_circuit,
    count_nonlocal_interactions,
    count_teleports,
    partition_circuit,
)
from hypergraph_partitioner.config import KAHYPAR_CONFIG
from hypergraph_partitioner.models.circuit_annotations import PartitionedCircuit
from tests.integration.simulation.statevector_test_utils import (
    INPUT_STATES,
    assert_statevectors_equivalent,
    embedded_original_to_qiskit,
    local_only_circuit,
    multi_segment_regression_circuit,
    remote_cz_circuit,
    simulate_statevector,
    teleport_regression_circuit,
    with_preparations,
)


def _rewrite_symbolic_for_qiskit(circuit: Circuit) -> Circuit:
    rewritten: list[InstructionType] = []
    for inst in circuit.instructions:
        if isinstance(inst, GateInstruction) and inst.name == "remote_cz":
            rewritten.append(
                GateInstruction(name="cz", qubits=list(inst.qubits), params=[], opaque=True)
            )
            continue
        if isinstance(inst, GateInstruction) and inst.name == "teleport":
            rewritten.append(
                GateInstruction(name="swap", qubits=list(inst.qubits), params=[], opaque=True)
            )
            continue
        rewritten.append(inst)
    return Circuit(qregs=circuit.qregs, cregs={}, instructions=rewritten)


def _annotated_distributed_to_qiskit(
    partitioned: PartitionedCircuit, qpu_data_capacity: int
) -> QuantumCircuit:
    distributed = annotated_to_distributed_circuit(partitioned, qpu_data_capacity=qpu_data_capacity)
    symbolic = _rewrite_symbolic_for_qiskit(distributed.as_monolithic_circuit())
    return CircuitConverters.to_qiskit(symbolic)


@pytest.mark.integration
@pytest.mark.parametrize("input_control", INPUT_STATES)
@pytest.mark.parametrize("input_target", INPUT_STATES)
def test_annotated_statevector_matches_original_for_remote_cz(
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

    annotated = simulate_statevector(_annotated_distributed_to_qiskit(partitioned, qpu_data_capacity=1))
    original = simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qpu_data_capacity=1)
    )

    assert_statevectors_equivalent(annotated, original)


@pytest.mark.integration
def test_annotated_statevector_matches_original_for_multi_segment_regression_circuit() -> None:
    circuit = multi_segment_regression_circuit()
    partitioned = partition_circuit(
        circuit,
        k=2,
        init_seg_size=10,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert len(partitioned.segments) == 5
    assert count_nonlocal_interactions(partitioned) == 14
    assert count_teleports(partitioned) == 8

    distributed = annotated_to_distributed_circuit(partitioned, qpu_data_capacity=4)
    names = [
        str(getattr(inst, "name", getattr(inst, "kind", "")))
        for inst in distributed.as_monolithic_circuit().instructions
    ]
    assert "teleport" in names
    assert "remote_cz" in names

    annotated = simulate_statevector(
        CircuitConverters.to_qiskit(_rewrite_symbolic_for_qiskit(distributed.as_monolithic_circuit()))
    )
    original = simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qpu_data_capacity=4)
    )

    assert_statevectors_equivalent(annotated, original)


@pytest.mark.integration
@pytest.mark.parametrize("input_control", INPUT_STATES)
@pytest.mark.parametrize("input_target", INPUT_STATES)
def test_annotated_statevector_matches_original_for_local_only(
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

    assert count_nonlocal_interactions(partitioned) == 0
    assert count_teleports(partitioned) == 0

    annotated = simulate_statevector(_annotated_distributed_to_qiskit(partitioned, qpu_data_capacity=2))
    original = simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qpu_data_capacity=2)
    )

    assert_statevectors_equivalent(annotated, original)


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
def test_annotated_statevector_matches_original_for_teleporting_circuit(
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

    distributed = annotated_to_distributed_circuit(partitioned, qpu_data_capacity=2)
    names = [
        str(getattr(inst, "name", getattr(inst, "kind", "")))
        for inst in distributed.as_monolithic_circuit().instructions
    ]
    assert "teleport" in names
    assert "remote_cz" in names

    annotated = simulate_statevector(CircuitConverters.to_qiskit(_rewrite_symbolic_for_qiskit(distributed.as_monolithic_circuit())))
    original = simulate_statevector(
        embedded_original_to_qiskit(circuit, partitioned, qpu_data_capacity=2)
    )

    assert_statevectors_equivalent(annotated, original)
