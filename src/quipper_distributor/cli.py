"""Command-line interface for quipper-distributor."""

from __future__ import annotations

import argparse
import sys

from quipper_distributor import config
from quipper_distributor.dist_circuit_builder import build_circuit
from quipper_distributor.hgraph_builder import build_hypergraph
from quipper_distributor.models.gate import is_cz
from quipper_distributor.parsing.quipper_ascii import emit_circuit, parse_circuit
from quipper_distributor.partitioner import partitioner
from quipper_distributor.preparation import prepare_circuit


def _count_non_local(segments) -> int:
    """Count non-local CZ interactions across all segments."""
    from quipper_distributor.models.gate import get_wires
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
        description="Distribute a Quipper circuit across K QPUs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-k", type=int, required=True, help="Number of QPUs (k > 1)")
    parser.add_argument("-s", type=int, required=True, help="QPU qubit capacity")
    parser.add_argument("-w", type=int, default=1000, metavar="INIT_SEG_SIZE",
                        help="Initial segment size (default: 1000)")
    parser.add_argument("-m", type=int, default=100, metavar="MAX_HEDGE_DIST",
                        help="Max hyperedge distance (default: 100)")
    parser.add_argument("--cc", action="store_true", help="Assume QPUs can execute CCZ gates")
    parser.add_argument("-d", default="./", metavar="DIR",
                        help="Directory for KaHyPar (default: ./)")
    parser.add_argument(
        "-o",
        default="gatecount",
        choices=["gatecount", "ascii"],
        help="Output format (default: gatecount)",
    )
    parser.add_argument("--save-trace", action="store_true", help="Save intermediate files")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output (default: True)")
    parser.add_argument("--quiet", action="store_true", help="Suppress partitioning progress")

    args = parser.parse_args()

    k = args.k
    seg_size = args.s  # QPU qubit capacity (used as init_seg_size in original)
    init_seg_size = args.w
    max_hedge_dist = args.m
    keep_ccz = args.cc
    cfg_dir = args.d
    output_format = args.o
    verbose = args.verbose and not args.quiet

    if k < 2 or seg_size < 1:
        print(
            "You must indicate the number of QPUs (k > 1) and their workspace qubit capacity (s > 0); "
            "example: quipper-distribute -k 2 -s 4",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read circuit from stdin
    circ_ascii = sys.stdin.read()
    circuit = parse_circuit(circ_ascii)

    n_qubits_total = k * seg_size
    n_wires = len(circuit.inputs)

    if n_qubits_total < n_wires:
        print(
            f"There are not enough qubits to run the circuit. Qubits required: {n_wires}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Preparation
    gates = prepare_circuit(circuit.gates, keep_ccz=keep_ccz)
    gate_count_input = len(gates)
    cz_count_input = sum(1 for g in gates if is_cz(g))

    # KaHyPar config path
    kahypar_config = config.KAHYPAR_CONFIG
    import os
    if not os.path.exists(kahypar_config):
        # Try relative to -d directory
        kahypar_config = os.path.join(cfg_dir, "kahypar/config/km1_kKaHyPar_sea20.ini")

    # Partition
    segs = partitioner(
        k=k,
        init_seg_size=init_seg_size,
        max_hedge_dist=max_hedge_dist,
        config_path=kahypar_config,
        n_qubits=n_wires,
        n_wires=n_wires,
        gates=gates,
    )

    # Build distributed circuit
    result_gates, new_wires, n_ebits, n_teleports = build_circuit(segs, n_wires, n_wires)

    # Output
    if output_format == "ascii":
        from quipper_distributor.models.circuit import Circuit, WireDecl, WireType
        out_circuit = Circuit(
            inputs=circuit.inputs,
            outputs=circuit.outputs,
            gates=result_gates,
        )
        print(emit_circuit(out_circuit))
    else:
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
