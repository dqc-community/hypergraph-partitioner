"""Unit tests for bosonic-model-native partition pipeline."""

from __future__ import annotations

from bosonic_converters import CircuitConverters
from bosonic_model.qasm import Translator
from qiskit.quantum_info import Operator

from hypergraph_partitioner.bosonic_pipeline import (
    _initial_segments,
    _preprocess,
    build_hypergraph_from_instructions,
    count_interactions,
    count_nonlocal_interactions,
    count_teleports,
    partition_circuit,
)
from hypergraph_partitioner.config import KAHYPAR_CONFIG
from hypergraph_partitioner.qiskit_normalization import normalize_to_one_qubit_and_cz


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

    segments = partition_circuit(
        circuit,
        k=2,
        init_seg_size=1000,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert len(segments) >= 1
    assert count_interactions(circuit.instructions) >= 2
    assert count_nonlocal_interactions(segments) >= 0
    assert count_teleports(segments, circuit.qubits()) >= 0


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
        k=2,
        max_hedge_dist=100,
        config_path=KAHYPAR_CONFIG,
    )

    assert len(segments) == 2
    assert count_interactions(segments[0].gates) == 1
    assert count_interactions(segments[1].gates) == 1
    assert segments[0].wire_range == (0, 0)
    assert segments[1].wire_range == (1, 1)


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

    preprocessed = _preprocess(circuit)

    assert [getattr(inst, "kind", None) for inst in preprocessed.instructions] == ["cz", "u", "u"]
    assert getattr(preprocessed.instructions[0], "qubits", None) == [0, 1]


def test_preprocess_step2_clusters_czs_into_a_single_hyperedge() -> None:
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
    preprocessed = _preprocess(circuit)

    step1_hyp = build_hypergraph_from_instructions(
        step1_only.instructions,
        n_qubits=circuit.qubits(),
        max_hedge_dist=100,
    )
    preprocessed_hyp = build_hypergraph_from_instructions(
        preprocessed.instructions,
        n_qubits=circuit.qubits(),
        max_hedge_dist=100,
    )

    assert [getattr(inst, "kind", None) for inst in preprocessed.instructions[:2]] == ["cz", "cz"]

    assert len(step1_hyp[0]) == 2
    assert all(len(hedge.wires) == 1 for hedge in step1_hyp[0])

    assert len(preprocessed_hyp[0]) == 1
    assert len(preprocessed_hyp[0][0].wires) == 2


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
    preprocessed = _preprocess(circuit)

    step1_op = Operator(CircuitConverters.to_qiskit(step1_only))
    preprocessed_op = Operator(CircuitConverters.to_qiskit(preprocessed))

    assert preprocessed_op.equiv(step1_op)
