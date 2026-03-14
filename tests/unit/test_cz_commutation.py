from __future__ import annotations

import math

from bosonic_model.instructions import CzInstruction, MeasureInstruction, ResetInstruction, UInstruction

from hypergraph_partitioner.cz_commutation import USemantics, classify_u, push_cz_early, z_u


def _u(wire: int, theta: float, phi: float, lam: float) -> UInstruction:
    params = [theta, phi, lam]
    return UInstruction(qubit=wire, qubits=[wire], theta=theta, phi=phi, lam=lam, params=params)


def _cz(control: int, target: int) -> CzInstruction:
    return CzInstruction(control=control, target=target, qubits=[control, target])


def test_classify_u_recognizes_supported_cases() -> None:
    assert classify_u(_u(0, 0.0, 0.2, 0.3)) == USemantics.Z_ROTATION
    assert classify_u(_u(0, math.pi, 0.0, math.pi)) == USemantics.X_GATE
    assert classify_u(_u(0, math.pi, math.pi / 2, math.pi / 2)) == USemantics.Y_GATE
    assert classify_u(_u(0, math.pi / 2, 0.0, math.pi)) == USemantics.H_GATE
    assert classify_u(_u(0, math.pi / 2, 0.0, 0.0)) == USemantics.OTHER


def test_z_rotation_commutes_through_cz() -> None:
    zrot = _u(0, 0.0, 0.2, 0.3)
    rewritten = push_cz_early([zrot, _cz(0, 1)])

    assert rewritten == [_cz(0, 1), zrot]


def test_x_commutes_and_creates_z_byproduct() -> None:
    x_gate = _u(0, math.pi, 0.0, math.pi)
    rewritten = push_cz_early([x_gate, _cz(0, 1)])

    assert rewritten == [_cz(0, 1), x_gate, z_u(1)]


def test_y_commutes_and_creates_z_byproduct() -> None:
    y_gate = _u(0, math.pi, math.pi / 2, math.pi / 2)
    rewritten = push_cz_early([y_gate, _cz(0, 1)])

    assert rewritten == [_cz(0, 1), y_gate, z_u(1)]


def test_h_blocks_commutation() -> None:
    h_gate = _u(0, math.pi / 2, 0.0, math.pi)
    original = [h_gate, _cz(0, 1)]

    assert push_cz_early(original) == original


def test_generic_u_blocks_commutation() -> None:
    other = _u(0, math.pi / 2, 0.0, 0.0)
    original = [other, _cz(0, 1)]

    assert push_cz_early(original) == original


def test_cz_moves_left_across_disjoint_u() -> None:
    disjoint = _u(2, math.pi, 0.0, math.pi)
    rewritten = push_cz_early([disjoint, _cz(0, 1)])

    assert rewritten == [_cz(0, 1), disjoint]


def test_byproducts_can_continue_commuting_rightward() -> None:
    x_gate = _u(0, math.pi, 0.0, math.pi)
    rewritten = push_cz_early([x_gate, _cz(0, 1), _cz(0, 2)])

    assert rewritten[:2] == [_cz(0, 1), _cz(0, 2)]
    assert rewritten[2] == x_gate
    assert rewritten[3:] == [z_u(2), z_u(1)]


def test_measure_and_reset_block_cz_movement() -> None:
    zrot = _u(0, 0.0, 0.0, math.pi / 4)
    measure = MeasureInstruction(qubit=1, cbit=0, qubits=[1])
    reset = ResetInstruction(qubit=1, qubits=[1])

    assert push_cz_early([zrot, measure, _cz(0, 1)]) == [zrot, measure, _cz(0, 1)]
    assert push_cz_early([zrot, reset, _cz(0, 1)]) == [zrot, reset, _cz(0, 1)]


def test_double_h_cancels() -> None:
    h_gate = _u(0, math.pi / 2, 0.0, math.pi)

    assert push_cz_early([h_gate, h_gate]) == []


def test_h_x_h_simplifies_to_z() -> None:
    h_gate = _u(0, math.pi / 2, 0.0, math.pi)
    x_gate = _u(0, math.pi, 0.0, math.pi)

    assert push_cz_early([h_gate, x_gate, h_gate]) == [z_u(0)]


def test_h_z_h_simplifies_to_x() -> None:
    h_gate = _u(0, math.pi / 2, 0.0, math.pi)
    z_gate = z_u(0)
    x_gate = _u(0, math.pi, 0.0, math.pi)

    assert push_cz_early([h_gate, z_gate, h_gate]) == [x_gate]


def test_push_cz_early_is_idempotent() -> None:
    instructions = [
        _u(0, math.pi / 2, 0.0, math.pi),
        _u(0, math.pi, 0.0, math.pi),
        _u(0, math.pi / 2, 0.0, math.pi),
        _cz(0, 1),
        _u(2, math.pi, 0.0, math.pi),
        _cz(2, 3),
    ]

    once = push_cz_early(instructions)
    twice = push_cz_early(once)

    assert twice == once


def test_multiple_rewrites_across_multiple_wires() -> None:
    zrot = _u(0, 0.0, 0.0, math.pi / 4)
    x_gate = _u(2, math.pi, 0.0, math.pi)
    y_gate = _u(3, math.pi, math.pi / 2, math.pi / 2)

    rewritten = push_cz_early(
        [
            zrot,
            _cz(0, 1),
            x_gate,
            _cz(2, 4),
            y_gate,
            _cz(3, 5),
        ]
    )

    assert rewritten == [
        _cz(0, 1),
        _cz(2, 4),
        _cz(3, 5),
        zrot,
        x_gate,
        z_u(4),
        y_gate,
        z_u(5),
    ]
