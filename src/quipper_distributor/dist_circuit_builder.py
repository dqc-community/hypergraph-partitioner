"""Distributed circuit synthesis.

Port of DCircBuilder.hs.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from quipper_distributor.models.gate import (
    CDiscard,
    Comment,
    Gate,
    QGate,
    QInit,
    QMeas,
    SignedWire,
    Wire,
)
from quipper_distributor.models.hypergraph import Hypergraph, Partition
from quipper_distributor.models.segment import Segment

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# (initial_hedge_pos, control_wire, target_wire, cz_pos, final_hedge_pos)
NonLocalConnection = tuple[int, Wire, Wire, int, int]

# sourceE := eDic[(ctrl, btarget)]  (a negative integer)
# sinkE   := sourceE - 1
EDic = dict[tuple[Wire, int], Wire]

# BindingFlags: True means the wire's qubit is currently allocated (live)
BindingFlags = dict[Wire, bool]


class EbitEntangler(BaseModel):
    kind: Literal["entangler"] = "entangler"
    key: tuple[Wire, int]  # (control_wire, target_block)
    ctrl_block: int
    pos: int


class EbitDisentangler(BaseModel):
    kind: Literal["disentangler"] = "disentangler"
    key: tuple[Wire, int]
    ctrl_block: int
    pos: int


EbitComponent = Annotated[Union[EbitEntangler, EbitDisentangler], Field(discriminator="kind")]


def _position(c: EbitComponent) -> int:
    return c.pos


def _is_entangler(c: EbitComponent) -> bool:
    return isinstance(c, EbitEntangler)


# ---------------------------------------------------------------------------
# updBindings
# ---------------------------------------------------------------------------


def _upd_bindings(gates: list[Gate], bindings: BindingFlags) -> BindingFlags:
    """Update binding flags based on which wires are created/consumed by gates."""
    b = dict(bindings)
    for g in gates:
        if isinstance(g, QInit):
            b[g.wire] = True
        elif isinstance(g, (QMeas,)):
            b[g.wire] = False
    return b


# ---------------------------------------------------------------------------
# non_local_connections
# ---------------------------------------------------------------------------


def non_local_connections(partition: Partition, hyp: Hypergraph) -> list[NonLocalConnection]:
    """Find all non-local CZ connections (wires in different blocks).

    Sorted by CZ position (ascending).
    """
    non_local_cs: list[NonLocalConnection] = []

    for v, hedges in hyp.items():
        for h in hedges:
            init_pos = h.nan
            out_pos = h.out_pos
            for w, cz_pos in h.wires:
                src_block = partition.get(v)
                snk_block = partition.get(w)
                if src_block is not None and snk_block is not None and src_block != snk_block:
                    non_local_cs.append((init_pos, v, w, cz_pos, out_pos))

    return sorted(non_local_cs, key=lambda c: c[3])


# ---------------------------------------------------------------------------
# ebit_info
# ---------------------------------------------------------------------------


def ebit_info(partition: Partition, nonlocal_cs: list[NonLocalConnection]) -> list[EbitComponent]:
    """Produce an ordered list of EbitEntangler/EbitDisentangler for each cut.

    Port of ebitInfo from DCircBuilder.hs.
    """
    # nub equivalent: unique (init_pos, ctrl, btarget, out_pos) tuples
    seen: set[tuple[int, Wire, int, int]] = set()
    e_info: list[tuple[int, Wire, int, int]] = []
    for init_pos, ctrl, snk, cz_pos, out_pos in nonlocal_cs:
        btarget = partition.get(snk, 0)
        key = (init_pos, ctrl, btarget, out_pos)
        if key not in seen:
            seen.add(key)
            e_info.append(key)

    components: list[EbitComponent] = []
    for init_pos, ctrl, btarget, out_pos in e_info:
        bctrl = partition.get(ctrl, 0)
        components.append(EbitEntangler(key=(ctrl, btarget), ctrl_block=bctrl, pos=init_pos))
        components.append(EbitDisentangler(key=(ctrl, btarget), ctrl_block=bctrl, pos=out_pos))

    # Sort: by position, then entanglers before disentanglers, then by key
    def sort_key(c: EbitComponent):
        return (c.pos, 0 if isinstance(c, EbitEntangler) else 1, c.key)

    return sorted(components, key=sort_key)


# ---------------------------------------------------------------------------
# distribute_czs
# ---------------------------------------------------------------------------


def _connect_cz_to(
    cs: list[NonLocalConnection], ws: list[Wire], partition: Partition, e_dic: EDic
) -> QGate:
    """Replace non-local control wires with ebit sink wires."""
    current_ws = list(ws)
    for c in cs:
        _, wire, sink, _, _ = c
        btarget = partition.get(sink, 0)
        source_e = e_dic.get((wire, btarget))
        if source_e is not None:
            ebit = source_e - 1  # sinkE = sourceE - 1
            current_ws = [ebit if w == wire else w for w in current_ws]
    target = current_ws[0]
    ctrl_wires = current_ws[1:]
    return QGate(
        name="CZ",
        inputs=[target],
        controls=[SignedWire(wire=w, positive=True) for w in ctrl_wires],
    )


def distribute_czs(
    nonlocal_cs: list[NonLocalConnection],
    gates: list[Gate],
    partition: Partition,
    e_dic: EDic,
) -> list[Gate]:
    """Replace non-local CZ control wires with ebit sink wires.

    Port of distributeCZs from DCircBuilder.hs.
    """
    if not nonlocal_cs:
        return list(gates)

    result: list[Gate] = []
    gate_list = list(gates)
    prev = 0
    cs_remaining = list(nonlocal_cs)

    while cs_remaining:
        c = cs_remaining[0]
        _, _, _, pos, _ = c
        # All connections at the same position
        this_cs = [c] + [x for x in cs_remaining[1:] if x[3] == pos]
        cs_remaining = cs_remaining[len(this_cs) :]

        result.extend(gate_list[prev:pos])
        g = gate_list[pos]
        # Get wires of this CZ gate
        if isinstance(g, QGate) and g.name == "CZ":
            ws = g.inputs + [sw.wire for sw in g.controls]
        else:
            ws = []
        g_prime = _connect_cz_to(this_cs, ws, partition, e_dic)
        result.append(g_prime)
        prev = pos + 1

    result.extend(gate_list[prev:])
    return result


# ---------------------------------------------------------------------------
# allocate_ebits
# ---------------------------------------------------------------------------


def _bell_gates(sink_e: Wire, source_e: Wire, b_sink: int, b_source: int) -> list[Gate]:
    return [
        QInit(value=False, wire=sink_e),
        QInit(value=False, wire=source_e),
        Comment(
            text="QPU_allocation",
            wire_labels=[(sink_e, f"{b_sink} ebit"), (source_e, f"{b_source} ebit")],
        ),
        QGate(name="bell", inputs=[sink_e, source_e]),
    ]


def _entangler_gates(
    source: Wire, source_e: Wire, sink_e: Wire, b_sink: int, b_source: int
) -> list[Gate]:
    """Ebit entangler gate sequence (matching Haskell's bell sequence)."""
    return _bell_gates(sink_e, source_e, b_sink, b_source) + [
        QGate(name="not", inputs=[source_e], controls=[SignedWire(wire=source, positive=True)]),
        QMeas(wire=source_e),
        QGate(name="X", inputs=[sink_e], controls=[SignedWire(wire=source_e, positive=True)]),
        Comment(text="QPU_allocation", wire_labels=[(source_e, "-1 ebit")]),
        CDiscard(wire=source_e),
    ]


def _disentangler_gates(source: Wire, sink_e: Wire) -> list[Gate]:
    """Ebit disentangler gate sequence."""
    return [
        QGate(name="H", inputs=[sink_e]),
        QMeas(wire=sink_e),
        QGate(name="Z", inputs=[source], controls=[SignedWire(wire=sink_e, positive=True)]),
        Comment(text="QPU_allocation", wire_labels=[(sink_e, "-1 ebit")]),
        CDiscard(wire=sink_e),
    ]


def allocate_ebits(
    components: list[EbitComponent],
    gates: list[Gate],
    e_dic: EDic,
) -> list[Gate]:
    """Insert Bell-pair and measurement gates at Entangler/Disentangler positions.

    Port of allocateEbits from DCircBuilder.hs.
    Note: components are processed from END to START (as in Haskell).
    """
    result = list(gates)
    prev = 0

    # Process in order (components are already sorted by position ascending)
    # We need to insert in a way that preserves offsets, so build incrementally
    output: list[Gate] = []
    gate_idx = 0

    for c in components:
        n = c.pos
        ((source, b_sink), b_source) = (c.key, c.ctrl_block)
        source_e = e_dic.get((source, b_sink))
        if source_e is None:
            continue
        sink_e = source_e - 1

        # Gates before position n
        output.extend(result[gate_idx:n])
        gate_idx = n

        if isinstance(c, EbitEntangler):
            output.extend(_entangler_gates(source, source_e, sink_e, b_sink, b_source))
        else:
            output.extend(_disentangler_gates(source, sink_e))

    # Remaining gates
    output.extend(result[gate_idx:])
    return output


# ---------------------------------------------------------------------------
# distribute_gates
# ---------------------------------------------------------------------------


def distribute_gates(
    gates: list[Gate], hyp: Hypergraph, partition: Partition
) -> tuple[list[Gate], int, int]:
    """Distribute a segment's gates, replacing non-local CZ wires with ebits.

    Returns (new_gates, new_wire_count, n_ebits).
    """
    non_local_cs = non_local_connections(partition, hyp)
    components = ebit_info(partition, non_local_cs)
    n_ebits = len(components) // 2

    # Build eDic: (ctrl_wire, target_block) → source_ebit_wire (negative integer)
    e_dic: EDic = {}
    wire_count = 0
    for c in components:
        if isinstance(c, EbitEntangler):
            key = c.key
            if key not in e_dic:
                wire_count += 2
                e_dic[key] = -(wire_count - 1)  # sourceE = -w-1 (negative)

    # Replace non-local CZ wires
    gates_with_czs = distribute_czs(non_local_cs, gates, partition, e_dic)

    # Insert ebit gates
    final_gates = allocate_ebits(components, gates_with_czs, e_dic)

    return final_gates, wire_count, n_ebits


# ---------------------------------------------------------------------------
# add_part_comments
# ---------------------------------------------------------------------------


def add_part_comments(
    bindings: BindingFlags, partition: Partition, gates: list[Gate]
) -> list[Gate]:
    """Add QPU allocation comments (matching addPartComments in DCircBuilder.hs)."""
    bound = [w for w, live in bindings.items() if live]
    part_comment = Comment(
        text="QPU_allocation",
        wire_labels=[(w, f"{partition.get(w, 0)} QPU") for w in sorted(bound)],
    )
    return [part_comment] + gates


# ---------------------------------------------------------------------------
# build_circuit
# ---------------------------------------------------------------------------


def build_circuit(
    segments: list[Segment], n_wires: int, n_inputs: int
) -> tuple[list[Gate], int, int, int]:
    """Synthesise the full distributed circuit from partitioned segments.

    Returns (gates, new_wire_count, n_ebits, n_teleports).
    """
    initial_bindings: BindingFlags = {w: True for w in range(n_inputs)}
    initial_bindings.update({w: False for w in range(n_inputs, n_wires)})

    return _build_circuit_rec(n_wires, initial_bindings, segments)


def _build_circuit_rec(
    n_wires: int, bindings: BindingFlags, segments: list[Segment]
) -> tuple[list[Gate], int, int, int]:
    """Recursive circuit building."""
    if not segments:
        return [], 0, 0, 0

    seg = segments[0]
    this_gates = seg.gates
    this_hyp = seg.hypergraph
    this_part = seg.partition

    dist_gates, this_extra_wires, this_ebits = distribute_gates(this_gates, this_hyp, this_part)
    dist_gates_commented = add_part_comments(bindings, this_part, dist_gates)

    if len(segments) == 1:
        return dist_gates_commented, this_extra_wires, this_ebits, 0

    next_part = segments[1].partition
    bindings_prime = _upd_bindings(this_gates, bindings)

    # Insert teleport gates for wires that change block and are currently alive
    tele_gates: list[Gate] = []
    for w in range(n_wires):
        if (
            bindings_prime.get(w, False)
            and w in this_part
            and w in next_part
            and this_part[w] != next_part[w]
        ):
            tele_gates.append(QGate(name="teleport", inputs=[w]))

    next_gates, next_wires, next_ebits, next_tps = _build_circuit_rec(
        n_wires, bindings_prime, segments[1:]
    )

    total_gates = dist_gates_commented + tele_gates + next_gates
    total_wires = max(this_extra_wires, next_wires)
    total_ebits = this_ebits + next_ebits
    total_tps = len(tele_gates) + next_tps

    return total_gates, total_wires, total_ebits, total_tps


def _upd_bindings(gates: list[Gate], bindings: BindingFlags) -> BindingFlags:
    """Update binding flags based on which wires are created/consumed by gates."""
    b = dict(bindings)
    for g in gates:
        if isinstance(g, QInit):
            b[g.wire] = True
        elif isinstance(g, QMeas):
            b[g.wire] = False
    return b
