from __future__ import annotations

from bosonic_converters import CircuitConverters
from bosonic_model import Circuit, ConditionalInstruction
from bosonic_model.instructions import InstructionType
from qiskit import transpile


def normalize_to_one_qubit_and_cz(circuit: Circuit) -> Circuit:
    """Normalize a bosonic circuit to 1-qubit gates plus CZ using Qiskit."""
    try:
        qiskit_circuit = CircuitConverters.to_qiskit(circuit)
        transpiled = transpile(
            qiskit_circuit,
            basis_gates=["u", "cz", "measure", "reset", "if_else"],
            initial_layout=list(range(qiskit_circuit.num_qubits)),
            layout_method="trivial",
            routing_method="none",
            optimization_level=3,
            translation_method="translator",
            seed_transpiler=0,
        )
        normalized = CircuitConverters.from_qiskit(transpiled)
        _validate_normalized_instructions(normalized.instructions)
        return normalized
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise ValueError(f"Failed to normalize circuit to 1-qubit + cz form: {exc}") from exc


def _validate_normalized_instructions(instructions: list[InstructionType]) -> None:
    allowed_kinds = {"u", "cz", "measure", "reset", "barrier"}
    for inst in instructions:
        if isinstance(inst, ConditionalInstruction):
            inner_kind = getattr(inst.op, "kind", None)
            if inner_kind not in allowed_kinds:
                raise ValueError(f"Unexpected conditional gate after normalization: {inner_kind}")
            continue

        kind = getattr(inst, "kind", None)
        if kind not in allowed_kinds:
            raise ValueError(f"Unexpected gate after normalization: {kind}")
