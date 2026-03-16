"""Protocol-building helpers for telegate and teledata lowerings."""

from __future__ import annotations

from bosonic_model import (
    Circuit,
    Condition,
    ConditionalInstruction,
    GateInstruction,
    MeasureInstruction,
    Register,
    ResetInstruction,
    RzzInstruction,
    UInstruction,
)
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from math import pi


def _registers(
    *, n_qubits: int, classical: tuple[tuple[str, int], ...]
) -> tuple[dict[str, Register], dict[str, Register]]:
    qregs = {"q": Register(name="q", size=n_qubits, base=0)}
    cregs: dict[str, Register] = {}
    base = 0
    for name, size in classical:
        cregs[name] = Register(name=name, size=size, base=base)
        base += size
    return qregs, cregs


def bell_pair_phi_plus_matrix() -> list[list[complex]]:
    scale = 1 / (2**0.5)
    return [
        [1 * scale, 0, 1 * scale, 0],
        [0, 1 * scale, 0, 1 * scale],
        [0, 1 * scale, 0, -1 * scale],
        [1 * scale, 0, -1 * scale, 0],
    ]


def build_telegate_remote_cz_dsl() -> Circuit:
    qregs, cregs = _registers(n_qubits=4, classical=(("c_start", 1), ("c_end", 1)))
    instructions: list = []

    data_ctrl = 0
    comm_ctrl = 1
    comm_tgt = 2
    data_tgt = 3
    c_start = 0
    c_end = 1

    instructions.append(
        GateInstruction(
            name="bell_pair_phi_plus",
            qubits=[comm_ctrl, comm_tgt],
            params=[],
            opaque=True,
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_ctrl,
            qubits=[comm_ctrl],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(
        UInstruction(
            qubit=data_ctrl,
            qubits=[data_ctrl],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_ctrl,
            qubits=[comm_ctrl],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        RzzInstruction(
            a=data_ctrl,
            b=comm_ctrl,
            qubits=[data_ctrl, comm_ctrl],
            theta=pi / 2,
            params=[pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_ctrl,
            qubits=[comm_ctrl],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(MeasureInstruction(qubit=comm_ctrl, cbit=c_start, qubits=[comm_ctrl]))
    instructions.append(ResetInstruction(qubit=comm_ctrl, qubits=[comm_ctrl]))
    instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_start, value=True),
            op=UInstruction(
                qubit=comm_tgt,
                qubits=[comm_tgt],
                theta=pi,
                phi=0,
                lam=pi,
                params=[pi, 0, pi],
            ),
            qubits=[comm_tgt],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_tgt,
            qubits=[comm_tgt],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=data_tgt,
            qubits=[data_tgt],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        RzzInstruction(
            a=comm_tgt,
            b=data_tgt,
            qubits=[comm_tgt, data_tgt],
            theta=pi / 2,
            params=[pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_tgt,
            qubits=[comm_tgt],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(MeasureInstruction(qubit=comm_tgt, cbit=c_end, qubits=[comm_tgt]))
    instructions.append(ResetInstruction(qubit=comm_tgt, qubits=[comm_tgt]))
    instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_end, value=True),
            op=UInstruction(
                qubit=data_ctrl,
                qubits=[data_ctrl],
                theta=0,
                phi=0,
                lam=pi,
                params=[0, 0, pi],
            ),
            qubits=[data_ctrl],
        )
    )

    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def build_ideal_remote_cz_dsl() -> Circuit:
    qregs, cregs = _registers(n_qubits=4, classical=())
    instructions: list = []
    instructions.append(GateInstruction(name="cz", qubits=[0, 3], params=[], opaque=True))
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def build_teledata_dsl() -> Circuit:
    qregs, cregs = _registers(n_qubits=3, classical=(("c_data", 1), ("c_comm", 1)))
    instructions: list = []

    data_src = 0
    comm_src = 1
    comm_dst = 2
    c_data = 0
    c_comm = 1

    instructions.append(
        GateInstruction(
            name="bell_pair_phi_plus",
            qubits=[comm_src, comm_dst],
            params=[],
            opaque=True,
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_src,
            qubits=[comm_src],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(
        UInstruction(
            qubit=data_src,
            qubits=[data_src],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_src,
            qubits=[comm_src],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    instructions.append(
        RzzInstruction(
            a=data_src,
            b=comm_src,
            qubits=[data_src, comm_src],
            theta=pi / 2,
            params=[pi / 2],
        )
    )
    instructions.append(
        UInstruction(
            qubit=comm_src,
            qubits=[comm_src],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(
        UInstruction(
            qubit=data_src,
            qubits=[data_src],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    instructions.append(MeasureInstruction(qubit=data_src, cbit=c_data, qubits=[data_src]))
    instructions.append(ResetInstruction(qubit=data_src, qubits=[data_src]))
    instructions.append(MeasureInstruction(qubit=comm_src, cbit=c_comm, qubits=[comm_src]))
    instructions.append(ResetInstruction(qubit=comm_src, qubits=[comm_src]))
    instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_comm, value=True),
            op=UInstruction(
                qubit=comm_dst,
                qubits=[comm_dst],
                theta=pi,
                phi=0,
                lam=pi,
                params=[pi, 0, pi],
            ),
            qubits=[comm_dst],
        )
    )
    instructions.append(
        ConditionalInstruction(
            condition=Condition(cbit=c_data, value=True),
            op=UInstruction(
                qubit=comm_dst,
                qubits=[comm_dst],
                theta=0,
                phi=0,
                lam=pi,
                params=[0, 0, pi],
            ),
            qubits=[comm_dst],
        )
    )
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def build_ideal_teledata_dsl() -> Circuit:
    qregs, cregs = _registers(n_qubits=3, classical=())
    return Circuit(qregs=qregs, cregs=cregs, instructions=[])


def to_aer_compatible_qiskit(circuit: QuantumCircuit) -> QuantumCircuit:
    rewritten = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for inst in circuit.data:
        op = inst.operation
        if op.name == "bell_pair_phi_plus":
            rewritten.append(
                UnitaryGate(bell_pair_phi_plus_matrix(), label=op.name),
                inst.qubits,
                inst.clbits,
            )
            continue
        if op.name in {"remote_link_psi_minus", "remote_link_psi_plus"}:
            rewritten.append(UnitaryGate(op.to_matrix(), label=op.name), inst.qubits, inst.clbits)
            continue
        rewritten.append(op, inst.qubits, inst.clbits)
    return rewritten
