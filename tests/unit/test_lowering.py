from __future__ import annotations

from math import pi

import pytest
from bosonic_converters import CircuitConverters
from bosonic_model import CzInstruction, DistributedCircuit, UInstruction
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity

from hypergraph_partitioner import lower_partitioned_circuit
from hypergraph_partitioner.lowering import to_aer_compatible_qiskit
from hypergraph_partitioner.models.annotated import (
    BlockId,
    BoundaryId,
    BoundaryTeleportOp,
    LocalOp,
    NonlocalCZOp,
    PartitionedCircuit,
    PartitionedSegment,
    SegmentId,
    WireId,
)


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


def test_lower_partitioned_circuit_lowers_local_ops_only() -> None:
    inst = _u(0, pi / 2, 0, pi)
    segment = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[inst],
        partition={WireId(0): BlockId(0)},
    )
    partitioned = PartitionedCircuit(
        segments=[segment],
        boundaries=[],
        operations=[LocalOp(segment_id=SegmentId(0), instruction=inst, blocks=(BlockId(0),))],
    )

    lowered = lower_partitioned_circuit(partitioned, qpu_data_capacity=1)

    assert isinstance(lowered, DistributedCircuit)
    assert lowered.qubits_per_node == {0: [0, 1, 2]}
    assert len(lowered.circuits[0].instructions) == 1
    lowered_inst = lowered.circuits[0].instructions[0]
    assert isinstance(lowered_inst, UInstruction)
    assert lowered_inst.qubit == 0


def test_lower_partitioned_circuit_telegate_matches_ideal_cz() -> None:
    prep0 = _prepare_label_ops(0, "+i")
    prep1 = _prepare_label_ops(1, "+")
    cz = CzInstruction(control=0, target=1, qubits=[0, 1])
    segment = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[*prep0, *prep1, cz],
        partition={WireId(0): BlockId(0), WireId(1): BlockId(1)},
    )
    ops = [
        *[
            LocalOp(segment_id=SegmentId(0), instruction=inst, blocks=(BlockId(0),))
            for inst in prep0
        ],
        *[
            LocalOp(segment_id=SegmentId(0), instruction=inst, blocks=(BlockId(1),))
            for inst in prep1
        ],
        NonlocalCZOp(
            segment_id=SegmentId(0),
            instruction=cz,
            control_wire=WireId(0),
            target_wire=WireId(1),
            control_block=BlockId(0),
            target_block=BlockId(1),
        ),
    ]
    partitioned = PartitionedCircuit(segments=[segment], boundaries=[], operations=ops)

    lowered = lower_partitioned_circuit(partitioned, qpu_data_capacity=1)
    remote = _simulate_density(CircuitConverters.to_qiskit(lowered.as_monolithic_circuit()))

    ideal = QuantumCircuit(6)
    for inst in prep0:
        if inst.qubit == 0:
            ideal.u(inst.theta, inst.phi, inst.lam, 0)
    for inst in prep1:
        if inst.qubit == 1:
            ideal.u(inst.theta, inst.phi, inst.lam, 3)
    ideal.cz(0, 3)
    ideal_density = _simulate_density(ideal)

    reduced_remote = partial_trace(remote, [1, 2, 4, 5])
    reduced_ideal = partial_trace(ideal_density, [1, 2, 4, 5])
    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)


def test_lower_partitioned_circuit_teledata_matches_ideal_state_transfer() -> None:
    prep = _prepare_label_ops(0, "+i")
    left = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=prep,
        partition={WireId(0): BlockId(0)},
    )
    right = PartitionedSegment(
        segment_id=SegmentId(1),
        instructions=[],
        partition={WireId(0): BlockId(1)},
    )
    ops = [
        *[
            LocalOp(segment_id=SegmentId(0), instruction=inst, blocks=(BlockId(0),))
            for inst in prep
        ],
        BoundaryTeleportOp(
            boundary_id=BoundaryId(0),
            wire=WireId(0),
            from_block=BlockId(0),
            to_block=BlockId(1),
        ),
    ]
    partitioned = PartitionedCircuit(segments=[left, right], boundaries=[], operations=ops)

    lowered = lower_partitioned_circuit(partitioned, qpu_data_capacity=1)
    remote = _simulate_density(CircuitConverters.to_qiskit(lowered.as_monolithic_circuit()))

    ideal = QuantumCircuit(6)
    for inst in prep:
        ideal.u(inst.theta, inst.phi, inst.lam, 5)
    ideal_density = _simulate_density(ideal)

    reduced_remote = partial_trace(remote, [0, 1, 2, 3, 4])
    reduced_ideal = partial_trace(ideal_density, [0, 1, 2, 3, 4])
    assert state_fidelity(reduced_remote, reduced_ideal) == pytest.approx(1.0, abs=1e-9)


def test_lower_partitioned_circuit_updates_wire_location_after_teledata() -> None:
    z_after = _u(0, 0, 0, pi)
    left = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[],
        partition={WireId(0): BlockId(0)},
    )
    right = PartitionedSegment(
        segment_id=SegmentId(1),
        instructions=[z_after],
        partition={WireId(0): BlockId(1)},
    )
    partitioned = PartitionedCircuit(
        segments=[left, right],
        boundaries=[],
        operations=[
            BoundaryTeleportOp(
                boundary_id=BoundaryId(0),
                wire=WireId(0),
                from_block=BlockId(0),
                to_block=BlockId(1),
            ),
            LocalOp(segment_id=SegmentId(1), instruction=z_after, blocks=(BlockId(1),)),
        ],
    )

    lowered = lower_partitioned_circuit(partitioned, qpu_data_capacity=1)
    node1_insts = lowered.circuits[1].instructions
    assert any(isinstance(inst, UInstruction) and inst.qubit == 5 and inst.lam == pi for inst in node1_insts)


def test_lower_partitioned_circuit_preserves_operation_order() -> None:
    prep = _u(0, pi / 2, 0, pi)
    post = _u(0, 0, 0, pi)
    left = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[prep],
        partition={WireId(0): BlockId(0)},
    )
    right = PartitionedSegment(
        segment_id=SegmentId(1),
        instructions=[post],
        partition={WireId(0): BlockId(1)},
    )
    partitioned = PartitionedCircuit(
        segments=[left, right],
        boundaries=[],
        operations=[
            LocalOp(segment_id=SegmentId(0), instruction=prep, blocks=(BlockId(0),)),
            BoundaryTeleportOp(
                boundary_id=BoundaryId(0),
                wire=WireId(0),
                from_block=BlockId(0),
                to_block=BlockId(1),
            ),
            LocalOp(segment_id=SegmentId(1), instruction=post, blocks=(BlockId(1),)),
        ],
    )

    lowered = lower_partitioned_circuit(partitioned, qpu_data_capacity=1)
    names = [
        str(getattr(inst, "name", getattr(inst, "kind", "")))
        for inst in lowered.as_monolithic_circuit().instructions
    ]
    assert names[0] == "u"
    assert "remote_bell_pair_phi_plus" in names[1:-1]
    assert names[-1] == "u"


def test_lower_partitioned_circuit_rejects_segments_exceeding_capacity() -> None:
    segment = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[],
        partition={WireId(0): BlockId(0), WireId(1): BlockId(0)},
    )
    partitioned = PartitionedCircuit(segments=[segment], boundaries=[], operations=[])

    with pytest.raises(ValueError, match="exceeding qpu_data_capacity"):
        lower_partitioned_circuit(partitioned, qpu_data_capacity=1)
