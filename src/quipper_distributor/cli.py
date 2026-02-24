"""Command-line interface for quipper-distributor."""

from __future__ import annotations

import argparse
import os
import sys

from bosonic_model.qasm import QasmError, Translator

from quipper_distributor import config
from quipper_distributor.bosonic_adapter import circuit_to_legacy_gates
from quipper_distributor.dist_circuit_builder import build_circuit
from quipper_distributor.models.gate import get_wires, is_cz
from quipper_distributor.partitioner import partitioner
from quipper_distributor.preparation import prepare_circuit


def _count_non_local(segments) -> int:
    """Count non-local CZ interactions across all segments."""
    total = 0
    for seg in segments:
        for g in seg.gates:
            if is_cz(g):
                wires = get_wires(g)
                blocks = set()
                for w in wires:
                    b = seg.partition.get(w)
                    if b is not None:
                        blocks.add(b)
                total += max(0, len(blocks) - 1)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="quipper-distribute",
        description="Distribute an OpenQASM circuit across K QPUs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-k", type=int, required=True, help="Number of QPUs (k > 1)")
    parser.add_argument("-s", type=int, required=True, help="QPU qubit capacity")
    parser.add_argument(
        "-w",
        type=int,
        default=1000,
        metavar="INIT_SEG_SIZE",
        help="Initial segment size (default: 1000)",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=100,
        metavar="MAX_HEDGE_DIST",
        help="Max hyperedge distance (default: 100)",
    )
    parser.add_argument("--cc", action="store_true", help="Assume QPUs can execute CCZ gates")
    parser.add_argument(
        "-d", default="./", metavar="DIR", help="Directory for KaHyPar (default: ./)"
    )
    parser.add_argument(
        "-o",
        default="gatecount",
        choices=["gatecount"],
        help="Output format (default: gatecount)",
    )
    parser.add_argument("--save-trace", action="store_true", help="Save intermediate files")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress partitioning progress")

    args = parser.parse_args()

    k = args.k
    seg_size = args.s
    init_seg_size = args.w
    max_hedge_dist = args.m
    keep_ccz = args.cc
    cfg_dir = args.d

    if k < 2 or seg_size < 1:
        print(
            "You must indicate the number of QPUs (k > 1) and their workspace qubit capacity (s > 0); "
            "example: quipper-distribute -k 2 -s 4",
            file=sys.stderr,
        )
        sys.exit(1)

    qasm_text = sys.stdin.read()
    try:
        circuit = Translator().from_qasm(qasm_text)
    except (QasmError, ValueError) as exc:
        print(f"Failed to parse OpenQASM input: {exc}", file=sys.stderr)
        sys.exit(1)

    n_qubits_total = k * seg_size
    n_wires = circuit.qubits()

    if n_qubits_total < n_wires:
        print(
            f"There are not enough qubits to run the circuit. Qubits required: {n_wires}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Phase 1 migration: convert to legacy gate model and reuse proven pipeline.
    input_gates = circuit_to_legacy_gates(circuit)
    gates = prepare_circuit(input_gates, keep_ccz=keep_ccz)
    gate_count_input = len(gates)
    cz_count_input = sum(1 for g in gates if is_cz(g))

    kahypar_config = config.KAHYPAR_CONFIG
    if not os.path.exists(kahypar_config):
        kahypar_config = os.path.join(cfg_dir, "kahypar/config/km1_kKaHyPar_sea20.ini")

    segs = partitioner(
        k=k,
        init_seg_size=init_seg_size,
        max_hedge_dist=max_hedge_dist,
        config_path=kahypar_config,
        n_qubits=n_wires,
        n_wires=n_wires,
        gates=gates,
    )

    _, _, n_ebits, n_teleports = build_circuit(segs, n_wires, n_wires)

    nonlocal_czs = _count_non_local(segs)

    print()
    print(f"Original gate count: {gate_count_input}")
    print(f"Original CZ count: {cz_count_input}")
    print(f"Original qubit count: {n_wires}")
    print()
    print(f"Number of nonlocal CZs: {nonlocal_czs}")
    print(f"Number of ebits due nonlocal CZs: {n_ebits}")
    print(f"Number of ebits due to teleportations: {n_teleports}")
    print(f"Total number of ebits: {n_ebits + n_teleports}")
    print()
    print(f"Extensions: {'KeepCCZ' if keep_ccz else 'N/A'}")
    print(f"#QPUs = {k}; QPU_size = {seg_size}")
    print(f"initSegSize = {init_seg_size}; maxHedgeDist = {max_hedge_dist}")
