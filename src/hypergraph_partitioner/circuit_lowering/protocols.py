"""Symbolic protocol builders and emitters for telegate and teledata."""

from __future__ import annotations

from math import pi

from bosonic_model import (
    Circuit,
    Condition,
    ConditionalInstruction,
    GateInstruction,
    InstructionType,
    MeasureInstruction,
    Register,
    ResetInstruction,
    RzzInstruction,
    UInstruction,
)


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


def build_telegate_remote_cz() -> Circuit:
    qregs, cregs = _registers(n_qubits=4, classical=(("c_start", 1), ("c_end", 1)))
    instructions: list[InstructionType] = []
    _emit_telegate_protocol(
        instructions,
        data_ctrl=0,
        comm_ctrl=1,
        comm_tgt=2,
        data_tgt=3,
        c_start=0,
        c_end=1,
    )
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def build_ideal_remote_cz() -> Circuit:
    qregs, cregs = _registers(n_qubits=4, classical=())
    instructions: list[InstructionType] = [
        GateInstruction(name="cz", qubits=[0, 3], params=[], opaque=True)
    ]
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def build_teledata() -> Circuit:
    qregs, cregs = _registers(n_qubits=3, classical=(("c_data", 1), ("c_comm", 1)))
    instructions: list[InstructionType] = []
    _emit_teledata_protocol(instructions, data_src=0, comm_src=1, comm_dst=2, c_data=0, c_comm=1)
    return Circuit(qregs=qregs, cregs=cregs, instructions=instructions)


def build_ideal_teledata() -> Circuit:
    qregs, cregs = _registers(n_qubits=3, classical=())
    return Circuit(qregs=qregs, cregs=cregs, instructions=[])


def _emit_telegate_protocol(
    instructions: list[InstructionType],
    *,
    data_ctrl: int,
    comm_ctrl: int,
    comm_tgt: int,
    data_tgt: int,
    c_start: int,
    c_end: int,
    include_bell: bool = True,
    target_instructions: list[InstructionType] | None = None,
    source_final_instructions: list[InstructionType] | None = None,
) -> None:
    target_instructions = target_instructions if target_instructions is not None else instructions
    source_final_instructions = (
        source_final_instructions if source_final_instructions is not None else instructions
    )
    if include_bell:
        instructions.append(
            GateInstruction(
                name="remote_link_phi_plus",
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
    target_instructions.append(
        ConditionalInstruction(
            condition=Condition(creg_base=c_start, creg_size=1, creg_value=1),
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
    target_instructions.append(
        UInstruction(
            qubit=comm_tgt,
            qubits=[comm_tgt],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    target_instructions.append(
        UInstruction(
            qubit=data_tgt,
            qubits=[data_tgt],
            theta=0,
            phi=0,
            lam=-pi / 2,
            params=[0, 0, -pi / 2],
        )
    )
    target_instructions.append(
        RzzInstruction(
            a=comm_tgt,
            b=data_tgt,
            qubits=[comm_tgt, data_tgt],
            theta=pi / 2,
            params=[pi / 2],
        )
    )
    target_instructions.append(
        UInstruction(
            qubit=comm_tgt,
            qubits=[comm_tgt],
            theta=pi / 2,
            phi=0,
            lam=pi,
            params=[pi / 2, 0, pi],
        )
    )
    target_instructions.append(MeasureInstruction(qubit=comm_tgt, cbit=c_end, qubits=[comm_tgt]))
    target_instructions.append(ResetInstruction(qubit=comm_tgt, qubits=[comm_tgt]))
    source_final_instructions.append(
        ConditionalInstruction(
            condition=Condition(creg_base=c_end, creg_size=1, creg_value=1),
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


def _emit_teledata_protocol(
    instructions: list[InstructionType],
    *,
    data_src: int,
    comm_src: int,
    comm_dst: int,
    c_data: int,
    c_comm: int,
    include_bell: bool = True,
    dest_instructions: list[InstructionType] | None = None,
) -> None:
    dest_instructions = dest_instructions if dest_instructions is not None else instructions
    if include_bell:
        instructions.append(
            GateInstruction(
                name="remote_link_phi_plus",
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
    dest_instructions.append(
        ConditionalInstruction(
            condition=Condition(creg_base=c_comm, creg_size=1, creg_value=1),
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
    dest_instructions.append(
        ConditionalInstruction(
            condition=Condition(creg_base=c_data, creg_size=1, creg_value=1),
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
