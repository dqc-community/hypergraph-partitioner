"""Parse and emit Quipper ASCII circuit format."""

from __future__ import annotations

import re

from quipper_distributor.models.circuit import Circuit, WireDecl, WireType
from quipper_distributor.models.gate import (
    CDiscard,
    CGate,
    CGateInv,
    CInit,
    CNot,
    Comment,
    CTerm,
    Gate,
    QDiscard,
    QGate,
    QInit,
    QMeas,
    QPrep,
    QRot,
    QTerm,
    QUnprep,
    SignedWire,
    Wire,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_wire_list(s: str) -> list[Wire]:
    """Parse '0, 1, 2' or '(0,1,2)' into a list of Wire ints."""
    s = s.strip().strip("()")
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_controls(s: str) -> list[SignedWire]:
    """Parse '[+0, -1, +2]' into a list of SignedWire."""
    s = s.strip().strip("[]")
    if not s:
        return []
    controls = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        positive = item[0] == "+"
        wire = int(item[1:])
        controls.append(SignedWire(wire=wire, positive=positive))
    return controls


def _parse_wire_decls(s: str) -> list[WireDecl]:
    """Parse '0:Qbit, 1:Cbit, 2:Qbit' into WireDecl list."""
    decls = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        wire_str, wtype_str = part.split(":")
        wtype = WireType.Qbit if wtype_str.strip() == "Qbit" else WireType.Cbit
        decls.append(WireDecl(wire=int(wire_str.strip()), wire_type=wtype))
    return decls


# ---------------------------------------------------------------------------
# Line parsers
# ---------------------------------------------------------------------------


def _parse_qgate(line: str) -> QGate | None:
    """QGate["name"](w1, w2) with controls=[+w3, -w4]"""
    m = re.match(r'^QGate\["([^"]+)"\]\*?\(([^)]*)\)(\s+with controls=\[([^\]]*)\])?', line)
    if not m:
        return None
    name = m.group(1)
    inputs_str = m.group(2)
    ctrl_str = m.group(4) or ""
    inputs = _parse_wire_list(inputs_str)
    controls = _parse_controls(ctrl_str)
    inverted = "*" in line[len('QGate["' + name + '"]') : line.index("(")]
    return QGate(name=name, inverted=inverted, inputs=inputs, controls=controls)


def _parse_qrot(line: str) -> QRot | None:
    """QRot["name",p1,p2,...](w) with controls=[...]"""
    m = re.match(r'^QRot\["([^"]+)"(,[^\]]*?)?\]\*?\(([^)]*)\)(\s+with controls=\[([^\]]*)\])?', line)
    if not m:
        return None
    name = m.group(1)
    params_str = m.group(2) or ""
    inputs_str = m.group(3)
    ctrl_str = m.group(5) or ""
    params = [float(p) for p in params_str.strip(",").split(",") if p.strip()]
    inputs = _parse_wire_list(inputs_str)
    controls = _parse_controls(ctrl_str)
    inverted = False
    return QRot(name=name, params=params, inverted=inverted, inputs=inputs, controls=controls)


def _parse_qinit(line: str) -> QInit | None:
    m = re.match(r"^QInit([01])\((\d+)\)", line)
    if not m:
        return None
    value = m.group(1) == "1"
    wire = int(m.group(2))
    return QInit(value=value, wire=wire)


def _parse_qterm(line: str) -> QTerm | None:
    m = re.match(r"^QTerm([01])\((\d+)\)", line)
    if not m:
        return None
    value = m.group(1) == "1"
    wire = int(m.group(2))
    return QTerm(value=value, wire=wire)


def _parse_qmeas(line: str) -> QMeas | None:
    m = re.match(r"^Measure\((\d+)\)", line)
    if not m:
        return None
    return QMeas(wire=int(m.group(1)))


def _parse_qdiscard(line: str) -> QDiscard | None:
    m = re.match(r"^QDiscard\((\d+)\)", line)
    if not m:
        return None
    return QDiscard(wire=int(m.group(1)))


def _parse_qprep(line: str) -> QPrep | None:
    m = re.match(r"^QPrep\((\d+)\)", line)
    if not m:
        return None
    return QPrep(wire=int(m.group(1)))


def _parse_qunprep(line: str) -> QUnprep | None:
    m = re.match(r"^QUnprep\((\d+)\)", line)
    if not m:
        return None
    return QUnprep(wire=int(m.group(1)))


def _parse_cinit(line: str) -> CInit | None:
    m = re.match(r"^CInit([01])\((\d+)\)", line)
    if not m:
        return None
    value = m.group(1) == "1"
    wire = int(m.group(2))
    return CInit(value=value, wire=wire)


def _parse_cterm(line: str) -> CTerm | None:
    m = re.match(r"^CTerm([01])\((\d+)\)", line)
    if not m:
        return None
    wire = int(m.group(2))
    return CTerm(wires=[wire], output=wire)


def _parse_cdiscard(line: str) -> CDiscard | None:
    m = re.match(r"^CDiscard\((\d+)\)", line)
    if not m:
        return None
    return CDiscard(wire=int(m.group(1)))


def _parse_cnot(line: str) -> CNot | None:
    m = re.match(r"^CNot\((\d+)\)", line)
    if not m:
        return None
    wire = int(m.group(1))
    return CNot(wire=wire, target=wire)


def _parse_cgate(line: str) -> CGate | None:
    m = re.match(r'^CGate\["([^"]+)"\]\(([^)]*)\)', line)
    if not m:
        return None
    name = m.group(1)
    wires = _parse_wire_list(m.group(2))
    output = wires[-1] if wires else 0
    inputs = wires[:-1]
    return CGate(name=name, inputs=inputs, output=output)


def _parse_cgateinv(line: str) -> CGateInv | None:
    m = re.match(r'^CGateInv\["([^"]+)"\]\(([^)]*)\)', line)
    if not m:
        return None
    name = m.group(1)
    wires = _parse_wire_list(m.group(2))
    output = wires[-1] if wires else 0
    inputs = wires[:-1]
    return CGateInv(name=name, inputs=inputs, output=output)


def _parse_comment(line: str) -> Comment | None:
    """Comment["text"](w1:"label1", w2:"label2")"""
    m = re.match(r'^Comment\["([^"]*)"\]\*?\(([^)]*)\)', line)
    if not m:
        return None
    text = m.group(1)
    body = m.group(2).strip()
    wire_labels: list[tuple[Wire, str]] = []
    if body:
        # Parse comma-separated  w:"label"  pairs
        for item in re.split(r',\s*(?=\d)', body):
            item = item.strip()
            wm = re.match(r'(\d+):"([^"]*)"', item)
            if wm:
                wire_labels.append((int(wm.group(1)), wm.group(2)))
    return Comment(text=text, wire_labels=wire_labels)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

_GATE_PARSERS = [
    _parse_qrot,  # Must come before _parse_qgate (longer prefix)
    _parse_qgate,
    _parse_qinit,
    _parse_qterm,
    _parse_qmeas,
    _parse_qdiscard,
    _parse_qprep,
    _parse_qunprep,
    _parse_cinit,
    _parse_cterm,
    _parse_cdiscard,
    _parse_cnot,
    _parse_cgate,
    _parse_cgateinv,
    _parse_comment,
]


def parse_circuit(text: str) -> Circuit:
    """Parse a Quipper ASCII circuit into a Circuit model."""
    inputs: list[WireDecl] = []
    outputs: list[WireDecl] = []
    gates: list[Gate] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Inputs:"):
            inputs = _parse_wire_decls(line[len("Inputs:"):].strip())
            continue
        if line.startswith("Outputs:"):
            outputs = _parse_wire_decls(line[len("Outputs:"):].strip())
            continue

        gate: Gate | None = None
        for parser in _GATE_PARSERS:
            gate = parser(line)
            if gate is not None:
                break

        if gate is not None:
            gates.append(gate)
        # Lines that don't match anything are silently ignored (e.g. blank / unknown)

    return Circuit(inputs=inputs, outputs=outputs, gates=gates)


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


def _emit_wire_decls(decls: list[WireDecl]) -> str:
    return ", ".join(f"{d.wire}:{d.wire_type.value}" for d in decls)


def _emit_controls(controls: list[SignedWire]) -> str:
    parts = [("+" if sw.positive else "-") + str(sw.wire) for sw in controls]
    return "[" + ", ".join(parts) + "]"


def emit_circuit(circuit: Circuit) -> str:
    """Emit a Circuit as Quipper ASCII text."""
    lines: list[str] = []
    lines.append("Inputs: " + _emit_wire_decls(circuit.inputs))

    for g in circuit.gates:
        if isinstance(g, QGate):
            inv = "*" if g.inverted else ""
            inputs_str = ", ".join(str(w) for w in g.inputs)
            line = f'QGate["{g.name}"]{inv}({inputs_str})'
            if g.controls:
                line += " with controls=" + _emit_controls(g.controls)
            lines.append(line)
        elif isinstance(g, QRot):
            params_str = "," + ",".join(str(p) for p in g.params) if g.params else ""
            inputs_str = ", ".join(str(w) for w in g.inputs)
            line = f'QRot["{g.name}"{params_str}]({inputs_str})'
            if g.controls:
                line += " with controls=" + _emit_controls(g.controls)
            lines.append(line)
        elif isinstance(g, QInit):
            lines.append(f"QInit{'1' if g.value else '0'}({g.wire})")
        elif isinstance(g, QTerm):
            lines.append(f"QTerm{'1' if g.value else '0'}({g.wire})")
        elif isinstance(g, QMeas):
            lines.append(f"Measure({g.wire})")
        elif isinstance(g, QDiscard):
            lines.append(f"QDiscard({g.wire})")
        elif isinstance(g, QPrep):
            lines.append(f"QPrep({g.wire})")
        elif isinstance(g, QUnprep):
            lines.append(f"QUnprep({g.wire})")
        elif isinstance(g, CInit):
            lines.append(f"CInit{'1' if g.value else '0'}({g.wire})")
        elif isinstance(g, CTerm):
            if g.wires:
                lines.append(f"CTerm{'1' if True else '0'}({g.wires[0]})")
        elif isinstance(g, CDiscard):
            lines.append(f"CDiscard({g.wire})")
        elif isinstance(g, CNot):
            lines.append(f"CNot({g.wire})")
        elif isinstance(g, CGate):
            wires = g.inputs + [g.output]
            lines.append(f'CGate["{g.name}"](' + ", ".join(str(w) for w in wires) + ")")
        elif isinstance(g, CGateInv):
            wires = g.inputs + [g.output]
            lines.append(f'CGateInv["{g.name}"](' + ", ".join(str(w) for w in wires) + ")")
        elif isinstance(g, Comment):
            body_parts = [f'{w}:"{lbl}"' for w, lbl in g.wire_labels]
            lines.append(f'Comment["{g.text}"](' + ", ".join(body_parts) + ")")

    lines.append("Outputs: " + _emit_wire_decls(circuit.outputs))
    return "\n".join(lines)
