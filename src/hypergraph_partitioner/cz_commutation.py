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
    """Repeatedly commute CZ gates left across supported U gates."""
    current = list(instructions)
    while True:
        updated, changed = _single_pass(current)
        if not changed:
            return updated
        current = updated


def _single_pass(instructions: list[InstructionType]) -> tuple[list[InstructionType], bool]:
    out: list[InstructionType] = []
    changed = False
    i = 0

    while i < len(instructions):
        if i + 1 < len(instructions):
            rewritten = _rewrite_adjacent_pair(instructions[i], instructions[i + 1])
            if rewritten is not None:
                out.extend(rewritten)
                i += 2
                changed = True
                continue

        out.append(instructions[i])
        i += 1

    return out, changed


def _rewrite_adjacent_pair(
    left: InstructionType, right: InstructionType
) -> list[InstructionType] | None:
    if isinstance(left, ConditionalInstruction) or isinstance(right, ConditionalInstruction):
        return None
    if not isinstance(right, CzInstruction):
        return None

    if isinstance(left, UInstruction):
        left_wire = left.qubit
        right_wires = set(right.qubits)

        if left_wire not in right_wires:
            return [right, left]

        semantics = classify_u(left)
        if semantics == USemantics.Z_ROTATION:
            return [right, left]
        if semantics in {USemantics.X_GATE, USemantics.Y_GATE}:
            other_wire = right.target if left_wire == right.control else right.control
            return [right, left, z_u(other_wire)]
        return None

    if isinstance(left, (MeasureInstruction, ResetInstruction)):
        return None

    # Conservatively block all other instruction kinds.
    return None


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
    params = [0.0, 0.0, math.pi]
    return UInstruction(
        qubit=wire,
        qubits=[wire],
        theta=params[0],
        phi=params[1],
        lam=params[2],
        params=params,
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
