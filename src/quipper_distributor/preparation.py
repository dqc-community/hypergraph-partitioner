"""Gate normalisation pipeline: translate circuit into CZ-normal form.

Direct port of Preparation.hs.
"""

from __future__ import annotations

from quipper_distributor.models.gate import (
    Gate,
    QGate,
    QInit,
    QMeas,
    QPrep,
    QRot,
    QTerm,
    QUnprep,
    SignedWire,
    Wire,
    is_classical,
    target_of,
)


# ---------------------------------------------------------------------------
# Step 1: separate_classical_control
# ---------------------------------------------------------------------------


def separate_classical_control(gates: list[Gate]) -> list[Gate]:
    """For QGate["not"/"X"] with mixed controls: split into classical/quantum.

    If all controls are on classical (Bit) wires this pass just keeps them.
    In our model after parsing, controls in QGate are always quantum.
    The separation relevant here: if a "not" gate has quantum controls only,
    keep as QGate["not"]; classical-only controls are already CDiscard/etc.
    This is mostly a no-op for well-formed Quipper ASCII input.
    """
    result: list[Gate] = []
    for g in gates:
        result.append(g)
    return result


# ---------------------------------------------------------------------------
# Step 2: remove_swaps
# ---------------------------------------------------------------------------


def remove_swaps(gates: list[Gate]) -> list[Gate]:
    """Replace QGate["swap"][q0, q1] with three CNOT gates."""
    result: list[Gate] = []
    for g in gates:
        if isinstance(g, QGate) and g.name == "swap" and len(g.inputs) == 2 and not g.controls:
            q0, q1 = g.inputs[0], g.inputs[1]
            result.append(QGate(name="not", inputs=[q0], controls=[SignedWire(wire=q1, positive=True)]))
            result.append(QGate(name="not", inputs=[q1], controls=[SignedWire(wire=q0, positive=True)]))
            result.append(QGate(name="not", inputs=[q0], controls=[SignedWire(wire=q1, positive=True)]))
        else:
            result.append(g)
    return result


# ---------------------------------------------------------------------------
# Step 3: decompose_toffoli
# ---------------------------------------------------------------------------

_T = lambda w: QGate(name="T", inputs=[w])
_T_inv = lambda w: QGate(name="T", inverted=True, inputs=[w])
_S = lambda w: QGate(name="S", inputs=[w])
_H = lambda w: QGate(name="H", inputs=[w])
_X = lambda w: QGate(name="X", inputs=[w])
_CNOT = lambda tgt, ctrl: QGate(name="not", inputs=[tgt], controls=[SignedWire(wire=ctrl, positive=True)])


def _toffoli_gates(target: Wire, c1: Wire, c2: Wire) -> list[Gate]:
    """Standard T-gate decomposition of a Toffoli gate (Selinger optimal)."""
    return [
        _H(target),
        _CNOT(target, c1),
        _T_inv(target),
        _CNOT(target, c2),
        _T(target),
        _CNOT(target, c1),
        _T_inv(target),
        _CNOT(target, c2),
        _T(target),
        _T(c1),
        _H(target),
        _CNOT(c1, c2),
        _T(c2),
        _T_inv(c1),
        _CNOT(c1, c2),
    ]


def decompose_toffoli(gates: list[Gate]) -> list[Gate]:
    """Decompose CCX (Toffoli) gates into T/CNOT primitives."""
    result: list[Gate] = []
    for g in gates:
        if (
            isinstance(g, QGate)
            and g.name == "not"
            and len(g.inputs) == 1
            and len(g.controls) == 2
        ):
            target = g.inputs[0]
            neg_ctrls = [sw for sw in g.controls if not sw.positive]
            pos_ctrls = [sw for sw in g.controls if sw.positive]

            # Wrap negative controls with X gates
            for sw in neg_ctrls:
                result.append(_X(sw.wire))

            c1 = g.controls[0].wire
            c2 = g.controls[1].wire
            result.extend(_toffoli_gates(target, c1, c2))

            for sw in neg_ctrls:
                result.append(_X(sw.wire))
        else:
            result.append(g)
    return result


# ---------------------------------------------------------------------------
# Step 4: to_controlled_z
# ---------------------------------------------------------------------------


def to_controlled_z(gates: list[Gate]) -> list[Gate]:
    """Convert QGate["not"][tgt] with 1 control into H-CZ-H."""
    result: list[Gate] = []
    for g in gates:
        if (
            isinstance(g, QGate)
            and g.name == "not"
            and len(g.inputs) == 1
            and len(g.controls) == 1
        ):
            target = g.inputs[0]
            ctrl = g.controls[0]
            result.append(QGate(name="H", inputs=[target]))
            result.append(QGate(name="CZ", inputs=[target], controls=[ctrl]))
            result.append(QGate(name="H", inputs=[target]))
        elif (
            isinstance(g, QRot)
            and g.controls
        ):
            # QRot with controls: treat as 2-qubit interaction for hypergraph purposes.
            # Wrap the target with H gates to produce a CZ-like hyperedge.
            target = g.inputs[0] if g.inputs else None
            if target is not None and len(g.controls) == 1:
                ctrl = g.controls[0]
                result.append(QGate(name="H", inputs=[target]))
                result.append(QGate(name="CZ", inputs=[target], controls=[ctrl]))
                result.append(QGate(name="H", inputs=[target]))
            else:
                result.append(g)
        else:
            result.append(g)
    return result


# ---------------------------------------------------------------------------
# Step 5: push_single_qubit_gates
# ---------------------------------------------------------------------------


def _name_of(g: Gate) -> str:
    if isinstance(g, QGate):
        return g.name
    return ""


def _gate_z(w: Wire, controls: list[SignedWire], ncf: bool) -> QGate:
    return QGate(name="Z", inputs=[w], controls=controls, ncf=ncf)


def _gate_x(w: Wire, controls: list[SignedWire], ncf: bool) -> QGate:
    return QGate(name="X", inputs=[w], controls=controls, ncf=ncf)


def _safe_head_is_h_no_ctrls(past: dict[Wire, list[Gate]], w: Wire) -> bool:
    stack = past.get(w, [])
    if not stack:
        return False
    head = stack[-1]
    return isinstance(head, QGate) and head.name == "H" and not head.controls


def _is_not_h(g: Gate) -> bool:
    return not (isinstance(g, QGate) and g.name == "H")


def _is_not_h_or_x(g: Gate) -> bool:
    if isinstance(g, QGate):
        return g.name not in ("H", "X")
    return True


def _add_byproducts(gates: list[Gate], w1: Wire, w2: Wire) -> list[Gate]:
    """When pushing an X gate through a CZ, a Z byproduct appears on the other wire."""
    result: list[Gate] = []
    for g in gates:
        if isinstance(g, QGate) and g.name == "X" and g.inputs == [w1] and not g.controls:
            byproduct = _gate_z(w2, g.controls, g.ncf)
            result.append(byproduct)
            result.append(g)
        elif isinstance(g, QGate) and g.name == "X" and g.inputs == [w2] and not g.controls:
            byproduct = _gate_z(w1, g.controls, g.ncf)
            result.append(byproduct)
            result.append(g)
        else:
            result.append(g)
    return result


def _add_x_for_neg_controls(wires: list[Wire]) -> list[Gate]:
    return [_X(w) for w in wires]


def _flush_wire(past: dict[Wire, list[Gate]], w: Wire) -> list[Gate]:
    """Return the accumulated gates for wire w (in circuit order) and clear them."""
    return list(reversed(past.get(w, [])))


def push_single_qubit_gates(gates: list[Gate]) -> list[Gate]:
    """Port of pushRec from Preparation.hs.

    Maintains a per-wire stack of single-qubit gates, flushing them when
    a multi-qubit or measurement gate is encountered.
    """
    # Collect all wires mentioned in the circuit
    all_wires: set[Wire] = set()
    for g in gates:
        if isinstance(g, QGate):
            all_wires.update(g.inputs)
            all_wires.update(sw.wire for sw in g.controls)
        elif isinstance(g, QRot):
            all_wires.update(g.inputs)
            all_wires.update(sw.wire for sw in g.controls)
        elif hasattr(g, "wire"):
            all_wires.add(g.wire)

    past: dict[Wire, list[Gate]] = {w: [] for w in all_wires}

    def append_to(w: Wire, gate: Gate) -> None:
        past.setdefault(w, []).append(gate)

    def flush_at(w: Wire) -> None:
        past[w] = []

    def gates_at(w: Wire) -> list[Gate]:
        return list(reversed(past.get(w, [])))

    result: list[Gate] = []
    remaining = list(gates)
    idx = 0

    while idx < len(remaining):
        g = remaining[idx]
        idx += 1

        if is_classical(g):
            result.append(g)
            continue

        if isinstance(g, QGate) and g.name == "CZ" and len(g.inputs) == 1 and len(g.controls) == 1:
            # Standard CZ gate (single target, single control)
            w = g.inputs[0]
            ctrl_signed = g.controls[0]
            ctrl = ctrl_signed.wire
            positive = ctrl_signed.positive

            w_past = past.get(w, [])
            ctrl_past = past.get(ctrl, [])

            # flushed = the part AFTER the H on each wire (i.e., the gates before H = non-H suffix reversed)
            # In the stack: last element is "top" (most recently added)
            # takeWhile isNotH from the top = take until we hit H
            def split_at_h(stack: list[Gate]) -> tuple[list[Gate], list[Gate]]:
                """Split stack at first H from top. Returns (non_h_top, rest_with_h)."""
                i = len(stack)
                while i > 0 and _is_not_h(stack[i - 1]):
                    i -= 1
                # stack[i:] are the top non-H gates, stack[:i] includes H and below
                return stack[i:], stack[:i]

            w_non_h, w_below = split_at_h(w_past)
            ctrl_non_h, ctrl_below = split_at_h(ctrl_past)

            # flushed = reverse of non-H part (these go into result)
            flushed_w = list(reversed(w_non_h))
            flushed_ctrl = list(reversed(ctrl_non_h))
            result.extend(flushed_w)
            result.extend(flushed_ctrl)

            # to_push = non-H parts (these get re-added to the front of remaining)
            # Actually in Haskell: toPush = addByproducts (concat $ map (reverse . takeWhile isNotH) wirePasts) (w,ctrl)
            # wirePasts are [past!w, past!ctrl]
            to_push_raw = list(reversed(w_non_h)) + list(reversed(ctrl_non_h))
            to_push = _add_byproducts(to_push_raw, w, ctrl)

            # flushedPast = clear w and ctrl
            past[w] = w_below
            past[ctrl] = ctrl_below

            result.append(g)

            # gateZNegCtrl
            gate_z_neg = []
            if not positive:
                gate_z_neg = [_gate_z(w, [], False)]

            remaining = gate_z_neg + to_push + remaining[idx:]
            idx = 0

        elif isinstance(g, QGate) and g.name == "CZ" and len(g.inputs) == 1 and len(g.controls) > 1:
            # CCZ gate (target + 2+ controls)
            w = g.inputs[0]
            ctrls = [sw.wire for sw in g.controls]
            all_wires_cz = [w] + ctrls

            def split_at_h_or_x(stack: list[Gate]) -> tuple[list[Gate], list[Gate]]:
                i = len(stack)
                while i > 0 and _is_not_h_or_x(stack[i - 1]):
                    i -= 1
                return stack[i:], stack[:i]

            flushed_parts = []
            to_push_parts = []
            for wire in all_wires_cz:
                wire_past = past.get(wire, [])
                non_hx, below = split_at_h_or_x(wire_past)
                flushed_parts.extend(reversed(non_hx))
                to_push_parts.extend(reversed(non_hx))
                past[wire] = below

            result.extend(flushed_parts)

            # gatesXNegCtrls: X gates for negative controls
            neg_ctrl_wires = [sw.wire for sw in g.controls if not sw.positive]
            gates_x_neg = _add_x_for_neg_controls(neg_ctrl_wires)
            result.extend(gates_x_neg)

            # g' with all positive controls
            g_prime = QGate(
                name="CZ",
                inputs=g.inputs,
                controls=[SignedWire(wire=sw.wire, positive=True) for sw in g.controls],
            )
            result.append(g_prime)

            remaining = gates_x_neg + to_push_parts + remaining[idx:]
            idx = 0

        elif isinstance(g, QGate) and g.name == "Z" and len(g.inputs) == 1:
            w = g.inputs[0]
            if _safe_head_is_h_no_ctrls(past, w):
                head_h = past[w][-1]
                past[w] = past[w][:-1]  # remove H from top
                append_to(w, head_h)
                append_to(w, _gate_x(w, g.controls, g.ncf))
            else:
                append_to(w, g)

        elif isinstance(g, QGate) and g.name == "X" and len(g.inputs) == 1 and not g.controls:
            w = g.inputs[0]
            if _safe_head_is_h_no_ctrls(past, w):
                head_h = past[w][-1]
                past[w] = past[w][:-1]
                append_to(w, head_h)
                append_to(w, _gate_z(w, g.controls, g.ncf))
            else:
                append_to(w, g)

        elif isinstance(g, QGate) and g.name == "Y" and len(g.inputs) == 1:
            w = g.inputs[0]
            if _safe_head_is_h_no_ctrls(past, w):
                head_h = past[w][-1]
                past[w] = past[w][:-1]
                append_to(w, head_h)
                append_to(w, _gate_z(w, g.controls, g.ncf))
                append_to(w, _gate_x(w, g.controls, g.ncf))
            else:
                # Y = iXZ → push Z then X
                append_to(w, _gate_x(w, g.controls, g.ncf))
                append_to(w, _gate_z(w, g.controls, g.ncf))

        elif isinstance(g, QGate) and g.name == "H" and len(g.inputs) == 1:
            w = g.inputs[0]
            if _safe_head_is_h_no_ctrls(past, w) and not g.controls:
                past[w] = past[w][:-1]  # Cancel HH
            else:
                append_to(w, g)

        elif isinstance(g, QGate) and g.name in ("S", "T") and len(g.inputs) == 1:
            append_to(g.inputs[0], g)

        elif isinstance(g, QGate):
            # Unrecognised multi-qubit gate or other variant — just emit
            result.append(g)

        elif isinstance(g, (QUnprep, QMeas)):
            wire = g.wire
            result.extend(gates_at(wire))
            flush_at(wire)
            result.append(g)

        elif hasattr(g, "wire") and isinstance(g.wire, int):
            wire = g.wire
            if isinstance(g, (QTerm,)):
                result.extend(gates_at(wire))
                flush_at(wire)
                result.append(g)
            elif isinstance(g, (QPrep, QInit)):
                result.append(g)
            else:
                result.append(g)

        elif isinstance(g, QRot):
            # QRot without controls: treat as single-qubit gate on first input wire
            if g.inputs:
                append_to(g.inputs[0], g)
            else:
                result.append(g)

        elif isinstance(g, Comment):
            pass  # Discard comments

        else:
            result.append(g)

    # Flush all remaining stacks
    for w in sorted(past.keys()):
        result.extend(gates_at(w))

    return result


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def prepare_circuit(gates: list[Gate], keep_ccz: bool = False) -> list[Gate]:
    """Run the full gate normalisation pipeline."""
    gates = separate_classical_control(gates)
    gates = remove_swaps(gates)
    if not keep_ccz:
        gates = decompose_toffoli(gates)
    gates = to_controlled_z(gates)
    gates = push_single_qubit_gates(gates)
    return gates
