from __future__ import annotations

from math import pi

import pytest
from bosonic_converters import CircuitConverters
from bosonic_model import Condition, ConditionalInstruction, CzInstruction, DistributedCircuit, GateInstruction, UInstruction
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity

from hypergraph_partitioner import (
    build_annotated_circuit,
    lower_distributed_circuit,
)
from hypergraph_partitioner.models.circuit_annotations import (
    NodeId,
    BoundaryId,
    PartitionedCircuit,
    PartitionedSegment,
    SegmentBoundary,
    TeleportBoundary,
    SegmentId,
    QubitId,
)
from tests.integration.simulation.statevector_test_utils import to_aer_compatible_qiskit


def _u(qubit: int, theta: float, phi: float, lam: float) -> UInstruction:
    return UInstruction(
        qubit=qubit,
        qubits=[qubit],
        theta=theta,
        phi=phi,
        lam=lam,
        params=[theta, phi, lam],
    )


def _prepare_label_ops(qubit: int, label: str) -> list[UInstruction]:
    if label == "0":
        return []
    if label == "1":
        return [_u(qubit, pi, 0, pi)]
    if label == "+":
        return [_u(qubit, pi / 2, 0, pi)]
    if label == "+i":
        return [_u(qubit, pi / 2, 0, pi), _u(qubit, 0, 0, pi / 2)]
    raise AssertionError(f"unexpected label {label}")


def _simulate_density(circuit: QuantumCircuit) -> DensityMatrix:
    simulator = AerSimulator(method="density_matrix")
    circuit = to_aer_compatible_qiskit(circuit)
    circuit.save_density_matrix()
    result = simulator.run(circuit).result()
    return DensityMatrix(result.data(0)["density_matrix"])


def test_build_annotated_circuit_converts_local_ops_only() -> None:
    inst = _u(0, pi / 2, 0, pi)
    segment = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[inst],
        partition={QubitId(0): NodeId(0)},
    )
    partitioned = PartitionedCircuit(
        segments=[segment],
        boundaries=[],
    )

    distributed = build_annotated_circuit(partitioned, qubits_per_node=1)

    assert isinstance(distributed, DistributedCircuit)
    assert distributed.qubits_per_node == {0: [0, 1, 2]}
    lowered_inst = distributed.circuits[0].instructions[0]
    assert isinstance(lowered_inst, UInstruction)
    assert lowered_inst.qubit == 0


def test_build_annotated_circuit_emits_remote_cz_symbolically() -> None:
    cz = CzInstruction(control=0, target=1, qubits=[0, 1])
    segment = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[cz],
        partition={QubitId(0): NodeId(0), QubitId(1): NodeId(1)},
    )
    partitioned = PartitionedCircuit(
        segments=[segment],
        boundaries=[],
    )

    distributed = build_annotated_circuit(partitioned, qubits_per_node=1)

    node0 = distributed.circuits[0].instructions
    node1 = distributed.circuits[1].instructions
    assert len(node0) == 1 and len(node1) == 1
    assert isinstance(node0[0], GateInstruction)
    assert node0[0].name == "remote_cz"
    assert node0[0] is node1[0]


def test_build_annotated_circuit_preserves_condition_on_remote_cz() -> None:
    conditional_cz = ConditionalInstruction(
        condition=Condition(creg_base=0, creg_size=1, creg_value=1),
        op=CzInstruction(control=0, target=1, qubits=[0, 1]),
    )
    segment = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[conditional_cz],
        partition={QubitId(0): NodeId(0), QubitId(1): NodeId(1)},
    )
    partitioned = PartitionedCircuit(
        segments=[segment],
        boundaries=[],
    )

    distributed = build_annotated_circuit(partitioned, qubits_per_node=1)

    node0 = distributed.circuits[0].instructions
    node1 = distributed.circuits[1].instructions
    assert len(node0) == 1 and len(node1) == 1
    assert isinstance(node0[0], ConditionalInstruction)
    assert node0[0].condition == conditional_cz.condition
    assert isinstance(node0[0].op, GateInstruction)
    assert node0[0].op.name == "remote_cz"
    assert node0[0] is node1[0]
    assert distributed.circuits[0].cregs["c"].size == 1


def test_build_annotated_circuit_emits_teleport_and_updates_destination() -> None:
    z_after = _u(0, 0, 0, pi)
    left = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[],
        partition={QubitId(0): NodeId(0)},
    )
    right = PartitionedSegment(
        segment_id=SegmentId(1),
        instructions=[z_after],
        partition={QubitId(0): NodeId(1)},
    )
    partitioned = PartitionedCircuit(
        segments=[left, right],
        boundaries=[
            SegmentBoundary(
                boundary_id=BoundaryId(0),
                left_segment_id=SegmentId(0),
                right_segment_id=SegmentId(1),
                teleports=[
                    TeleportBoundary(
                        qubit=QubitId(0),
                        from_node=NodeId(0),
                        to_node=NodeId(1),
                    )
                ],
            )
        ],
    )

    distributed = build_annotated_circuit(partitioned, qubits_per_node=1)

    node0 = distributed.circuits[0].instructions
    node1 = distributed.circuits[1].instructions
    assert isinstance(node0[0], GateInstruction) and node0[0].name == "teleport"
    assert node1[0] is node0[0]
    assert any(isinstance(inst, UInstruction) and inst.qubit == 5 for inst in node1[1:])


def test_build_annotated_circuit_reuses_receiver_slots_after_back_and_forth_teleports() -> None:
    """
    regression ref: https://github.com/dqc-community/hypergraph-partitioner/pull/1/changes#r2949105997
    """
    segments = [
        PartitionedSegment(
            segment_id=SegmentId(0),
            instructions=[],
            partition={QubitId(0): NodeId(0)},
        ),
        PartitionedSegment(
            segment_id=SegmentId(1),
            instructions=[],
            partition={QubitId(0): NodeId(1)},
        ),
        PartitionedSegment(
            segment_id=SegmentId(2),
            instructions=[],
            partition={QubitId(0): NodeId(0)},
        ),
        PartitionedSegment(
            segment_id=SegmentId(3),
            instructions=[],
            partition={QubitId(0): NodeId(1)},
        ),
    ]
    partitioned = PartitionedCircuit(
        segments=segments,
        boundaries=[
            SegmentBoundary(
                boundary_id=BoundaryId(0),
                left_segment_id=SegmentId(0),
                right_segment_id=SegmentId(1),
                teleports=[
                    TeleportBoundary(
                        qubit=QubitId(0),
                        from_node=NodeId(0),
                        to_node=NodeId(1),
                    )
                ],
            ),
            SegmentBoundary(
                boundary_id=BoundaryId(1),
                left_segment_id=SegmentId(1),
                right_segment_id=SegmentId(2),
                teleports=[
                    TeleportBoundary(
                        qubit=QubitId(0),
                        from_node=NodeId(1),
                        to_node=NodeId(0),
                    )
                ],
            ),
            SegmentBoundary(
                boundary_id=BoundaryId(2),
                left_segment_id=SegmentId(2),
                right_segment_id=SegmentId(3),
                teleports=[
                    TeleportBoundary(
                        qubit=QubitId(0),
                        from_node=NodeId(0),
                        to_node=NodeId(1),
                    )
                ],
            ),
        ],
    )

    distributed = build_annotated_circuit(partitioned, qubits_per_node=1)

    names = [
        str(getattr(inst, "name", getattr(inst, "kind", "")))
        for inst in distributed.as_monolithic_circuit().instructions
    ]
    assert names == ["teleport", "teleport", "teleport"]


def test_build_annotated_circuit_preserves_operation_order() -> None:
    prep = _u(0, pi / 2, 0, pi)
    post = _u(0, 0, 0, pi)
    left = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[prep],
        partition={QubitId(0): NodeId(0)},
    )
    right = PartitionedSegment(
        segment_id=SegmentId(1),
        instructions=[post],
        partition={QubitId(0): NodeId(1)},
    )
    partitioned = PartitionedCircuit(
        segments=[left, right],
        boundaries=[
            SegmentBoundary(
                boundary_id=BoundaryId(0),
                left_segment_id=SegmentId(0),
                right_segment_id=SegmentId(1),
                teleports=[
                    TeleportBoundary(
                        qubit=QubitId(0),
                        from_node=NodeId(0),
                        to_node=NodeId(1),
                    )
                ],
            )
        ],
    )

    distributed = build_annotated_circuit(partitioned, qubits_per_node=1)
    names = [
        str(getattr(inst, "name", getattr(inst, "kind", "")))
        for inst in distributed.as_monolithic_circuit().instructions
    ]
    assert names == ["u", "teleport", "u"]


def test_build_annotated_circuit_rejects_segments_exceeding_capacity() -> None:
    segment = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[],
        partition={QubitId(0): NodeId(0), QubitId(1): NodeId(0)},
    )
    partitioned = PartitionedCircuit(segments=[segment], boundaries=[])

    with pytest.raises(ValueError, match="exceeding qubits_per_node"):
        build_annotated_circuit(partitioned, qubits_per_node=1)


def test_lower_distributed_circuit_telegate_matches_ideal_cz() -> None:
    prep0 = _prepare_label_ops(0, "+i")
    prep1 = _prepare_label_ops(1, "+")
    cz = CzInstruction(control=0, target=1, qubits=[0, 1])
    segment = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[*prep0, *prep1, cz],
        partition={QubitId(0): NodeId(0), QubitId(1): NodeId(1)},
    )
    partitioned = PartitionedCircuit(segments=[segment], boundaries=[])

    symbolic = build_annotated_circuit(partitioned, qubits_per_node=1)
    lowered = lower_distributed_circuit(symbolic)
    remote = _simulate_density(CircuitConverters.to_qiskit(lowered.as_monolithic_circuit()))

    ideal = QuantumCircuit(6)
    for inst in prep0:
        ideal.u(inst.theta, inst.phi, inst.lam, 0)
    for inst in prep1:
        ideal.u(inst.theta, inst.phi, inst.lam, 3)
    ideal.cz(0, 3)
    ideal_density = _simulate_density(ideal)

    reduced_remote = partial_trace(remote, [1, 2, 4, 5])
    reduced_ideal = partial_trace(ideal_density, [1, 2, 4, 5])
    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)


def test_lower_distributed_circuit_teledata_matches_ideal_state_transfer() -> None:
    prep = _prepare_label_ops(0, "+i")
    left = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=prep,
        partition={QubitId(0): NodeId(0)},
    )
    right = PartitionedSegment(
        segment_id=SegmentId(1),
        instructions=[],
        partition={QubitId(0): NodeId(1)},
    )
    partitioned = PartitionedCircuit(
        segments=[left, right],
        boundaries=[
            SegmentBoundary(
                boundary_id=BoundaryId(0),
                left_segment_id=SegmentId(0),
                right_segment_id=SegmentId(1),
                teleports=[
                    TeleportBoundary(
                        qubit=QubitId(0),
                        from_node=NodeId(0),
                        to_node=NodeId(1),
                    )
                ],
            )
        ],
    )

    symbolic = build_annotated_circuit(partitioned, qubits_per_node=1)
    lowered = lower_distributed_circuit(symbolic)
    remote = _simulate_density(CircuitConverters.to_qiskit(lowered.as_monolithic_circuit()))

    ideal = QuantumCircuit(6)
    for inst in prep:
        ideal.u(inst.theta, inst.phi, inst.lam, 5)
    ideal_density = _simulate_density(ideal)

    reduced_remote = partial_trace(remote, [0, 1, 2, 3, 4])
    reduced_ideal = partial_trace(ideal_density, [0, 1, 2, 3, 4])
    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)
