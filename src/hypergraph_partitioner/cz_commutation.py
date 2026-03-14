from __future__ import annotations

import math
from enum import Enum

from bosonic_model import ConditionalInstruction
from bosonic_model.instructions import CzInstruction, InstructionType, MeasureInstruction, ResetInstruction, UInstruction

TAU = 2 * math.pi
ANGLE_TOL = 1e-9


class USemantics(str, Enum):
    Z_ROTATION = "z_rotation"
    X_GATE = "x_gate"
    Y_GATE = "y_gate"
    H_GATE = "h_gate"
    OTHER = "other"


def push_cz_early(instructions: list[InstructionType]) -> list[InstructionType]:
    """Commute CZ gates left via per-wire buffering of supported U gates."""
    out: list[InstructionType] = []
    pending: list[UInstruction] = []

    for inst in instructions:
        if isinstance(inst, ConditionalInstruction):
            out.extend(pending)
            pending = []
            out.append(inst)
            continue

        if isinstance(inst, UInstruction):
            pending = _append_pending(pending, inst)
            continue

        if isinstance(inst, CzInstruction):
            flushed, pending = _advance_across_cz(pending, inst)
            out.extend(flushed)
            out.append(inst)
            continue

        if isinstance(inst, (MeasureInstruction, ResetInstruction)):
            out.extend(pending)
            pending = []
            out.append(inst)
            continue

        qubits = set(getattr(inst, "qubits", []) or [])
        if qubits:
            flushed, pending = _flush_wires(pending, qubits)
            out.extend(flushed)
        else:
            out.extend(pending)
            pending = []
        out.append(inst)

    out.extend(pending)
    return out


def classify_u(inst: UInstruction) -> USemantics:
    theta = _normalize_angle(inst.theta)
    phi = _normalize_angle(inst.phi)
    lam = _normalize_angle(inst.lam)

    if _angle_close(theta, 0.0):
        return USemantics.Z_ROTATION
    if _angle_close(theta, math.pi) and _angle_close(phi, 0.0) and _angle_close(lam, math.pi):
        return USemantics.X_GATE
    if _angle_close(theta, math.pi) and _angle_close(phi, math.pi / 2) and _angle_close(
        lam, math.pi / 2
    ):
        return USemantics.Y_GATE
    if _angle_close(theta, math.pi / 2) and _angle_close(phi, 0.0) and _angle_close(
        lam, math.pi
    ):
        return USemantics.H_GATE
    return USemantics.OTHER


def z_u(wire: int) -> UInstruction:
    return _u_gate(wire, 0.0, 0.0, math.pi)


def x_u(wire: int) -> UInstruction:
    return _u_gate(wire, math.pi, 0.0, math.pi)


def h_u(wire: int) -> UInstruction:
    return _u_gate(wire, math.pi / 2, 0.0, math.pi)


def _u_gate(wire: int, theta: float, phi: float, lam: float) -> UInstruction:
    params = [theta, phi, lam]
    return UInstruction(
        qubit=wire,
        qubits=[wire],
        theta=theta,
        phi=phi,
        lam=lam,
        params=params,
    )


def _append_pending(pending: list[UInstruction], inst: UInstruction) -> list[UInstruction]:
    updated = list(pending)
    updated.append(inst)

    while True:
        same_wire = [i for i, gate in enumerate(updated) if gate.qubit == inst.qubit]
        if len(same_wire) < 2:
            return updated

        prev_idx, last_idx = same_wire[-2], same_wire[-1]
        rewritten = _rewrite_same_wire_pair(updated[prev_idx], updated[last_idx])
        if rewritten is None:
            return updated

        if not rewritten:
            updated = [gate for i, gate in enumerate(updated) if i not in {prev_idx, last_idx}]
        else:
            new_updated: list[UInstruction] = []
            for i, gate in enumerate(updated):
                if i == prev_idx:
                    new_updated.append(rewritten[0])
                elif i == last_idx:
                    if len(rewritten) == 2:
                        new_updated.append(rewritten[1])
                else:
                    new_updated.append(gate)
            updated = new_updated


def _rewrite_same_wire_pair(left: UInstruction, right: UInstruction) -> list[UInstruction] | None:
    wire = left.qubit
    left_semantics = classify_u(left)
    right_semantics = classify_u(right)

    if left_semantics == USemantics.H_GATE and right_semantics == USemantics.H_GATE:
        return []
    if left_semantics == USemantics.H_GATE and _is_exact_x(right):
        return [z_u(wire), h_u(wire)]
    if left_semantics == USemantics.H_GATE and _is_exact_z(right):
        return [x_u(wire), h_u(wire)]
    return None


def _advance_across_cz(
    pending: list[UInstruction], cz: CzInstruction
) -> tuple[list[UInstruction], list[UInstruction]]:
    involved_wires = set(cz.qubits)
    crossable_indices = _crossable_suffix_indices(pending, involved_wires)

    flushed: list[UInstruction] = []
    remaining: list[tuple[int, UInstruction]] = []
    for idx, gate in enumerate(pending):
        if gate.qubit in involved_wires and idx not in crossable_indices:
            flushed.append(gate)
        else:
            remaining.append((idx, gate))

    next_pending: list[UInstruction] = []
    for idx, gate in remaining:
        next_pending.append(gate)
        if idx in crossable_indices and classify_u(gate) in {USemantics.X_GATE, USemantics.Y_GATE}:
            other_wire = cz.target if gate.qubit == cz.control else cz.control
            next_pending.append(z_u(other_wire))

    return flushed, next_pending


def _crossable_suffix_indices(pending: list[UInstruction], involved_wires: set[int]) -> set[int]:
    crossable: set[int] = set()
    for wire in involved_wires:
        indices = [i for i, gate in enumerate(pending) if gate.qubit == wire]
        for idx in reversed(indices):
            if _is_pushable(classify_u(pending[idx])):
                crossable.add(idx)
            else:
                break
    return crossable


def _flush_wires(
    pending: list[UInstruction], wires: set[int]
) -> tuple[list[UInstruction], list[UInstruction]]:
    flushed: list[UInstruction] = []
    remaining: list[UInstruction] = []
    for gate in pending:
        if gate.qubit in wires:
            flushed.append(gate)
        else:
            remaining.append(gate)
    return flushed, remaining


def _is_pushable(semantics: USemantics) -> bool:
    return semantics in {USemantics.Z_ROTATION, USemantics.X_GATE, USemantics.Y_GATE}


def _is_exact_x(inst: UInstruction) -> bool:
    return classify_u(inst) == USemantics.X_GATE


def _is_exact_z(inst: UInstruction) -> bool:
    return (
        _angle_close(_normalize_angle(inst.theta), 0.0)
        and _angle_close(_normalize_angle(inst.phi), 0.0)
        and _angle_close(_normalize_angle(inst.lam), math.pi)
    )


def _normalize_angle(angle: float) -> float:
    normalized = math.fmod(angle, TAU)
    if normalized < 0:
        normalized += TAU
    if _angle_close(normalized, TAU):
        return 0.0
    return normalized


def _angle_close(a: float, b: float) -> bool:
    diff = math.fmod(a - b, TAU)
    if diff > math.pi:
        diff -= TAU
    if diff < -math.pi:
        diff += TAU
    return abs(diff) <= ANGLE_TOL
