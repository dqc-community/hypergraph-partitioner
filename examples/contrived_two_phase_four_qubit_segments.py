"""Contrived four-qubit example to probe seam splitting and merging.

Phase 1 repeatedly couples qubits (0,1) and (2,3).
Phase 2 repeatedly couples qubits (0,3) and (1,2).

The intent is to create a sharp change in the interaction graph around the
midpoint and see whether the full ``partition_circuit(...)`` pipeline preserves
multiple final segments.
"""

from __future__ import annotations

import argparse
import logging

from bosonic_model.qasm import Translator

from hypergraph_partitioner import count_nonlocal_interactions, count_teleports, partition_circuit
from hypergraph_partitioner.config import KAHYPAR_CONFIG


def _append_phase_one_pattern(lines: list[str]) -> None:
    lines.append("h q[0];")
    lines.append("cz q[0], q[1];")
    lines.append("t q[2];")
    lines.append("cz q[2], q[3];")


def _append_phase_two_pattern(lines: list[str]) -> None:
    lines.append("h q[0];")
    lines.append("cz q[0], q[3];")
    lines.append("t q[1];")
    lines.append("cz q[1], q[2];")


def _build_qasm(*, phase_one_steps: int, phase_two_steps: int) -> str:
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "qreg q[4];",
    ]

    for _ in range(phase_one_steps):
        _append_phase_one_pattern(lines)

    for _ in range(phase_two_steps):
        _append_phase_two_pattern(lines)

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-one-steps", type=int, default=50)
    parser.add_argument("--phase-two-steps", type=int, default=50)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--init-seg-size", type=int, default=7)
    parser.add_argument("--max-hedge-dist", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("hypergraph_partitioner.partitioner").setLevel(logging.DEBUG)
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
    logging.getLogger("qiskit.compiler").setLevel(logging.WARNING)

    qasm = _build_qasm(
        phase_one_steps=args.phase_one_steps,
        phase_two_steps=args.phase_two_steps,
    )
    circuit = Translator().from_qasm(qasm)
    result = partition_circuit(
        circuit,
        k=args.k,
        init_seg_size=args.init_seg_size,
        max_hedge_dist=args.max_hedge_dist,
        config_path=KAHYPAR_CONFIG,
    )

    print("Input summary")
    print(
        {
            "phase_one_steps": args.phase_one_steps,
            "phase_two_steps": args.phase_two_steps,
            "k": args.k,
            "init_seg_size": args.init_seg_size,
            "max_hedge_dist": args.max_hedge_dist,
        }
    )
    print()

    print("PartitionedCircuit summary")
    print(f"segments={len(result.segments)}")
    print(f"boundaries={len(result.boundaries)}")
    print(f"nonlocal_czs={count_nonlocal_interactions(result)}")
    print(f"teleports={count_teleports(result)}")
    print()

    for segment in result.segments:
        print(
            f"segment {segment.segment_id}: "
            f"instructions={len(segment.instructions)} partition={segment.partition}"
        )

    if result.boundaries:
        print()
        for boundary in result.boundaries:
            print(
                f"boundary {boundary.boundary_id}: "
                f"{boundary.left_segment_id}->{boundary.right_segment_id} "
                f"teleports={boundary.teleports}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
