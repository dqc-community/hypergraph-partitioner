"""Adapter from bosonic_model circuits/instructions to legacy partitioner gate models.

This module intentionally keeps the existing partitioning pipeline stable while
migrating input semantics to OpenQASM + bosonic_model.
"""

from __future__ import annotations

from bosonic_model import (
    BarrierInstruction,
    Circuit,
    ConditionalInstruction,
    GateInstruction,
    MeasureInstruction,
    ResetInstruction,
)

from quipper_distributor.models.gate import Gate, QGate, QInit, QMeas, QRot, SignedWire

_SINGLE_QUBIT_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
}


def _as_qgate_from_generic(name: str, qubits: list[int], params: list[float]) -> Gate | None:
    lower = name.lower()

    if lower == "cx" and len(qubits) == 2:
        ctrl, tgt = qubits[0], qubits[1]
        return QGate(name="not", inputs=[tgt], controls=[SignedWire(wire=ctrl, positive=True)])

    if lower == "cz" and len(qubits) == 2:
        ctrl, tgt = qubits[0], qubits[1]
        return QGate(name="CZ", inputs=[tgt], controls=[SignedWire(wire=ctrl, positive=True)])

    if lower == "ccx" and len(qubits) == 3:
        c1, c2, tgt = qubits[0], qubits[1], qubits[2]
        return QGate(
            name="not",
            inputs=[tgt],
            controls=[
                SignedWire(wire=c1, positive=True),
                SignedWire(wire=c2, positive=True),
            ],
        )

    if lower in _SINGLE_QUBIT_MAP and len(qubits) == 1:
        return QGate(name=_SINGLE_QUBIT_MAP[lower], inputs=[qubits[0]])

    if lower in {"rx", "ry", "rz", "u", "u1", "u2", "u3", "p"} and len(qubits) == 1:
        return QRot(name=name, params=list(params), inputs=[qubits[0]])

    if lower == "swap" and len(qubits) == 2:
        return QGate(name="swap", inputs=[qubits[0], qubits[1]])

    if len(qubits) == 1:
        return QGate(name=name, inputs=[qubits[0]])

    if len(qubits) >= 2:
        target = qubits[-1]
        ctrls = [SignedWire(wire=w, positive=True) for w in qubits[:-1]]
        return QGate(name=name, inputs=[target], controls=ctrls)

    return None


def instruction_to_legacy_gate(inst: object) -> Gate | None:
    """Convert a bosonic_model instruction to a legacy gate model object.

    Returns None for instructions that should be ignored by the current
    partitioning flow (e.g., barriers).
    """
    if isinstance(inst, BarrierInstruction):
        return None

    if isinstance(inst, ConditionalInstruction):
        # Phase 1 stats pipeline ignores classical conditions and conservatively
        # projects the operation itself into the interaction stream.
        return instruction_to_legacy_gate(inst.op)

    if isinstance(inst, GateInstruction):
        return _as_qgate_from_generic(inst.name, list(inst.qubits), list(inst.params))

    if isinstance(inst, MeasureInstruction):
        return QMeas(wire=inst.qubit)

    if isinstance(inst, ResetInstruction):
        return QInit(value=False, wire=inst.qubit)

    # Generic fallback for typed instruction classes that expose name/qubits/params,
    # e.g. CcxInstruction, CxInstruction, UInstruction, etc.
    name = getattr(inst, "name", None)
    qubits = getattr(inst, "qubits", None)
    params = getattr(inst, "params", None)
    if isinstance(name, str) and isinstance(qubits, list):
        safe_params = list(params) if isinstance(params, list) else []
        return _as_qgate_from_generic(name, list(qubits), safe_params)

    # Unknown instruction variants are ignored in phase 1 stats mode.
    return None


def circuit_to_legacy_gates(circuit: Circuit) -> list[Gate]:
    gates: list[Gate] = []
    for inst in circuit.instructions:
        gate = instruction_to_legacy_gate(inst)
        if gate is not None:
            gates.append(gate)
    return gates
