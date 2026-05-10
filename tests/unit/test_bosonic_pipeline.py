"""Unit tests for bosonic-model-native partition pipeline."""

from __future__ import annotations

import random

from bosonic_converters import CircuitConverters
from bosonic_model import DistributedCircuit
from bosonic_model.instructions import CzInstruction, UInstruction
from bosonic_model.qasm import Translator
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from hypergraph_partitioner.bosonic_pipeline import (
    _annotate_partitioned_circuit,
    _build_hypergraph_from_instructions,
    _count_interactions,
    _count_nonlocal_interactions,
    _count_teleports,
    _partition_to_partitioned_circuit,
    _preprocess,
    _initial_segments,
    iter_annotated_operations,
    partition_circuit,
)
from hypergraph_partitioner.segment_merger import ignore_last_seam
from hypergraph_partitioner.preprocessing.cz_commutation import push_cz_early
from hypergraph_partitioner.models.circuit_annotations import (
    BoundaryTeleportOp,
    NonlocalCZOp,
    SegmentBoundary,
)
from hypergraph_partitioner.preprocessing.normalization import normalize_to_one_qubit_and_cz
from hypergraph_partitioner.models.segment import SeamStop, Segment


def _assert_normalized_to_one_qubit_and_cz(circuit) -> None:
    allowed_single = {
        "u",
        "measure",
        "reset",
        "barrier",
        "conditional",
    }
    for inst in circuit.instructions:
        kind = getattr(inst, "kind", None)
        if kind == "conditional":
            assert getattr(inst.op, "kind", None) in {"u", "cz", "measure", "reset", "barrier"}
            continue
        if len(getattr(inst, "qubits", []) or []) >= 2:
            assert kind == "cz"
        else:
            assert kind in allowed_single


def _random_supported_qiskit_circuit(
    *,
    n_qubits: int,
    depth: int,
    seed: int,
    basis_gates: tuple[str, ...],
) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)

    one_qubit_ops = {
        "h": lambda q: qc.h(q),
        "x": lambda q: qc.x(q),
        "y": lambda q: qc.y(q),
        "z": lambda q: qc.z(q),
        "s": lambda q: qc.s(q),
        "t": lambda q: qc.t(q),
    }
    two_qubit_ops = {
        "cx": lambda a, b: qc.cx(a, b),
        "cz": lambda a, b: qc.cz(a, b),
        "swap": lambda a, b: qc.swap(a, b),
    }

    for _ in range(depth):
        gate = rng.choice(basis_gates)
        if gate in one_qubit_ops:
            one_qubit_ops[gate](rng.randrange(n_qubits))
            continue
        if gate in two_qubit_ops:
            a, b = rng.sample(range(n_qubits), 2)
            two_qubit_ops[gate](a, b)
            continue
        if gate == "ccx":
            a, b, c = rng.sample(range(n_qubits), 3)
            qc.ccx(a, b, c)
            continue
        raise AssertionError(f"Unexpected random test gate: {gate}")

    return qc


@pytest.mark.integration
def test_partition_circuit_runs_on_small_qasm() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        h q[0];
        cx q[0], q[1];
        ccx q[0], q[1], q[2];
        """
    )

    result = partition_circuit(
        circuit,
        nodes=2,
        init_seg_size=1000,
        max_hedge_dist=100,
        qubits_per_node=2,
    )

    assert isinstance(result, DistributedCircuit)
    assert result.circuits
    assert result.qubits_per_node
    assert _count_interactions(circuit.instructions) >= 2


@pytest.mark.integration
def test_partition_circuit_can_return_lowered_distributed_circuit() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """
    )

    result = partition_circuit(
        circuit,
        nodes=2,
        init_seg_size=1000,
        max_hedge_dist=100,
        qubits_per_node=1,
        output="lowered",
    )

    names = [
        str(getattr(inst, "name", getattr(inst, "kind", "")))
        for inst in result.as_monolithic_circuit().instructions
    ]
    assert isinstance(result, DistributedCircuit)
    assert "remote_cz" not in names
    assert "teleport" not in names


@pytest.mark.integration
def test_partition_circuit_rejects_unknown_output_mode() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cz q[0], q[1];
        """
    )

    with pytest.raises(ValueError, match="unsupported output mode"):
        partition_circuit(
            circuit,
            nodes=2,
            init_seg_size=1000,
            max_hedge_dist=100,
            qubits_per_node=1,
            output="partitioned",
        )


def test_annotated_circuit_orders_boundary_ops_between_segments() -> None:
    cz = CzInstruction(control=0, target=1, qubits=[0, 1])
    u = UInstruction(qubit=0, qubits=[0], theta=0.0, phi=0.0, lam=0.0, params=[0.0, 0.0, 0.0])
    segments = [
        Segment(gates=[cz], partition={0: 0, 1: 1}, seam=SeamStop(), segment_range=(0, 0)),
        Segment(gates=[u], partition={0: 1, 1: 1}, seam=SeamStop(), segment_range=(1, 1)),
    ]

    result = _annotate_partitioned_circuit(segments)

    assert len(result.segments) == 2
    assert len(result.boundaries) == 1
    assert isinstance(result.boundaries[0], SegmentBoundary)
    operations = list(iter_annotated_operations(result))

    first_boundary_idx = next(
        i for i, op in enumerate(operations) if isinstance(op, BoundaryTeleportOp)
    )
    first_nonlocal_idx = next(
        i for i, op in enumerate(operations) if isinstance(op, NonlocalCZOp)
    )

    assert first_nonlocal_idx < first_boundary_idx


@pytest.mark.integration
def test_initial_segments_splits_by_interaction_count() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        h q[0];
        cx q[0], q[1];
        h q[2];
        cx q[1], q[2];
        """
    )

    segments = _initial_segments(
        circuit.instructions,
        init_seg_size=1,
        n_qubits=circuit.qubits(),
        nodes=2,
        max_hedge_dist=100,
    )

    assert len(segments) == 2
    assert _count_interactions(segments[0].gates) == 1
    assert _count_interactions(segments[1].gates) == 1
    assert segments[0].segment_range == (0, 0)
    assert segments[1].segment_range == (1, 1)


def test_normalize_to_one_qubit_and_cz_rewrites_cx() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cx q[0], q[1];
        """
    )

    normalized = normalize_to_one_qubit_and_cz(circuit)

    _assert_normalized_to_one_qubit_and_cz(normalized)
    assert all(getattr(inst, "kind", None) != "cx" for inst in normalized.instructions)


def test_normalize_to_one_qubit_and_cz_rewrites_swap() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        swap q[0], q[1];
        """
    )

    normalized = normalize_to_one_qubit_and_cz(circuit)

    _assert_normalized_to_one_qubit_and_cz(normalized)
    assert all(getattr(inst, "kind", None) != "swap" for inst in normalized.instructions)


def test_normalize_to_one_qubit_and_cz_rewrites_ccx() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        ccx q[0], q[1], q[2];
        """
    )

    normalized = normalize_to_one_qubit_and_cz(circuit)

    _assert_normalized_to_one_qubit_and_cz(normalized)
    assert all(getattr(inst, "kind", None) != "ccx" for inst in normalized.instructions)


def test_normalize_to_one_qubit_and_cz_is_deterministic() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        h q[0];
        cx q[0], q[1];
        ccx q[0], q[1], q[2];
        """
    )

    normalized_a = normalize_to_one_qubit_and_cz(circuit)
    normalized_b = normalize_to_one_qubit_and_cz(circuit)

    assert [inst.model_dump() for inst in normalized_a.instructions] == [
        inst.model_dump() for inst in normalized_b.instructions
    ]


def test_preprocess_step2_pulls_cz_earlier_for_supported_u() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q[0];
        cz q[0], q[1];
        """
    )

    res = normalize_to_one_qubit_and_cz(circuit)
    preprocessed = push_cz_early(res.instructions)

    assert [getattr(inst, "kind", None) for inst in preprocessed] == ["cz", "u", "u"]
    assert getattr(preprocessed[0], "qubits", None) == [0, 1]


def test_preprocess_step2_keeps_qubit_interactions_in_circuit_order() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        x q[0];
        cz q[0], q[1];
        z q[0];
        cz q[0], q[2];
        """
    )

    step1_only = normalize_to_one_qubit_and_cz(circuit)
    preprocessed = push_cz_early(step1_only.instructions)

    step1_hyp = _build_hypergraph_from_instructions(
        step1_only.instructions,
        n_qubits=circuit.qubits(),
    )
    preprocessed_hyp = _build_hypergraph_from_instructions(
        preprocessed,
        n_qubits=circuit.qubits(),
    )

    assert len(step1_hyp.interactions) == 2
    assert len(preprocessed_hyp.interactions) == 2
    assert step1_hyp.qubit_to_interactions[0] == [0, 1]
    assert preprocessed_hyp.qubit_to_interactions[0] == [0, 1]


def test_preprocess_step2_preserves_circuit_unitary() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        x q[0];
        cz q[0], q[1];
        z q[0];
        cz q[0], q[2];
        """
    )

    step1_only = normalize_to_one_qubit_and_cz(circuit)
    preprocessed = push_cz_early(step1_only.instructions)

    step1_op = Operator(CircuitConverters.to_qiskit(step1_only))
    preprocessed_op = Operator(CircuitConverters.to_qiskit(step1_only.model_copy(update={"instructions": preprocessed})))

    assert preprocessed_op.equiv(step1_op)


@pytest.mark.parametrize(
    ("n_qubits", "depth", "seed", "basis_gates"),
    [
        (3, 14, 7, ("h", "x", "z", "cx", "cz")),
        (4, 20, 11, ("h", "s", "t", "cx", "cz", "swap")),
        (5, 24, 23, ("x", "y", "z", "h", "cx", "cz", "swap", "ccx")),
        (6, 28, 31, ("h", "x", "y", "z", "s", "t", "cx", "cz", "swap", "ccx")),
    ],
)
def test_preprocess_step2_preserves_unitary_for_random_supported_circuits(
    n_qubits: int,
    depth: int,
    seed: int,
    basis_gates: tuple[str, ...],
) -> None:
    qiskit_circuit = _random_supported_qiskit_circuit(
        n_qubits=n_qubits,
        depth=depth,
        seed=seed,
        basis_gates=basis_gates,
    )
    circuit = CircuitConverters.from_qiskit(qiskit_circuit)

    step1_only = normalize_to_one_qubit_and_cz(circuit)
    preprocessed = push_cz_early(step1_only.instructions)

    step1_op = Operator(CircuitConverters.to_qiskit(step1_only))
    preprocessed_op = Operator(CircuitConverters.to_qiskit(step1_only.model_copy(update={"instructions": preprocessed})))

    assert preprocessed_op.equiv(step1_op)


def test_annotated_circuit_marks_nonlocal_czs_and_boundary_teleports() -> None:
    cz = CzInstruction(control=0, target=1, qubits=[0, 1])
    u = UInstruction(qubit=0, qubits=[0], theta=0.0, phi=0.0, lam=0.0, params=[0.0, 0.0, 0.0])
    segments = [
        Segment(gates=[cz], partition={0: 0, 1: 1}, seam=SeamStop(), segment_range=(0, 0)),
        Segment(gates=[u], partition={0: 1, 1: 1}, seam=SeamStop(), segment_range=(1, 1)),
    ]

    result = _annotate_partitioned_circuit(segments)

    assert _count_nonlocal_interactions(result) == 1
    assert _count_teleports(result) == 1
    operations = list(iter_annotated_operations(result))
    assert any(isinstance(op, NonlocalCZOp) for op in operations)
    assert any(isinstance(op, BoundaryTeleportOp) for op in operations)


@pytest.mark.integration
def test_partition_circuit_end_to_end_annotates_nonlocal_czs() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[6];
        h q[1];
        cx q[0], q[1];
        h q[3];
        cx q[2], q[3];
        cx q[0], q[3];
        h q[5];
        cx q[4], q[5];
        h q[1];
        h q[4];
        cx q[1], q[4];
        h q[3];
        h q[4];
        h q[5];
        """
    )

    result = _partition_to_partitioned_circuit(
        circuit,
        nodes=2,
        init_seg_size=1,
        max_hedge_dist=100,
    )

    nonlocal_ops = [op for op in iter_annotated_operations(result) if isinstance(op, NonlocalCZOp)]

    assert len(result.segments) >= 1
    assert len(result.boundaries) == max(0, len(result.segments) - 1)
    assert len(nonlocal_ops) >= 1
    assert all(op.control_node != op.target_node for op in nonlocal_ops)


@pytest.mark.integration
def test_real_circuit_initial_segments_annotate_multiple_segments_and_teleports() -> None:
    circuit = Translator().from_qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[6];
        cx q[0], q[1];
        cx q[0], q[2];
        cx q[0], q[1];
        x q[0];
        z q[1];
        cx q[0], q[4];
        cx q[0], q[5];
        cx q[0], q[4];
        """
    )

    normalized = _preprocess(circuit)
    initial = _initial_segments(
        normalized.instructions,
        init_seg_size=1,
        n_qubits=circuit.qubits(),
        nodes=2,
        max_hedge_dist=100,
    )
    result = _annotate_partitioned_circuit(ignore_last_seam(initial))

    teleport_ops = [
        op for op in iter_annotated_operations(result) if isinstance(op, BoundaryTeleportOp)
    ]
    assert len(result.segments) >= 2
    assert len(result.boundaries) == len(result.segments) - 1
    assert len(teleport_ops) >= 1
    assert sum(len(boundary.teleports) for boundary in result.boundaries) == len(teleport_ops)
