"""Search for a deterministic circuit that survives as multiple final segments.

This example is diagnostic. It enables DEBUG logs from ``merge_min(...)`` so the
seam-merging decisions are visible while searching candidate circuits.
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Iterable

from bosonic_converters import CircuitConverters
from qiskit import QuantumCircuit

from hypergraph_partitioner import count_nonlocal_interactions, count_teleports, partition_circuit
from hypergraph_partitioner.config import KAHYPAR_CONFIG


logger = logging.getLogger(__name__)


def _random_supported_qiskit_circuit(
    *,
    n_qubits: int,
    depth: int,
    seed: int,
    basis_gates: tuple[str, ...],
) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)

    one_qubit_ops = {
        "h": lambda q: qc.h(q),
        "x": lambda q: qc.x(q),
        "y": lambda q: qc.y(q),
        "z": lambda q: qc.z(q),
        "s": lambda q: qc.s(q),
        "t": lambda q: qc.t(q),
    }
    two_qubit_ops = {
        "cx": lambda a, b: qc.cx(a, b),
        "cz": lambda a, b: qc.cz(a, b),
        "swap": lambda a, b: qc.swap(a, b),
    }

    for _ in range(depth):
        gate = rng.choice(basis_gates)
        if gate in one_qubit_ops:
            one_qubit_ops[gate](rng.randrange(n_qubits))
            continue
        if gate in two_qubit_ops:
            a, b = rng.sample(range(n_qubits), 2)
            two_qubit_ops[gate](a, b)
            continue
        if gate == "ccx":
            a, b, c = rng.sample(range(n_qubits), 3)
            qc.ccx(a, b, c)
            continue
        raise AssertionError(f"Unexpected gate: {gate}")

    return qc


def _cluster_stage_circuit(*, n_qubits: int, repeats: int) -> QuantumCircuit:
    """Build a deterministic staged circuit with sharply changing connectivity."""
    qc = QuantumCircuit(n_qubits)
    mid = n_qubits // 2

    # Stage 1: first-half hub on q0.
    for _ in range(repeats):
        for target in range(1, mid):
            qc.cz(0, target)
        qc.h(0)

    # Stage 2: cross-half interactions through the same hub.
    for _ in range(repeats):
        for target in range(mid, n_qubits):
            qc.cz(0, target)
        qc.s(0)

    # Stage 3: second-half hub on the last qubit.
    for _ in range(repeats):
        for source in range(mid, n_qubits - 1):
            qc.cz(source, n_qubits - 1)
        qc.h(n_qubits - 1)

    return qc


def _candidate_circuits() -> Iterable[tuple[str, dict[str, int], QuantumCircuit]]:
    for n_qubits in (8, 10, 12):
        for repeats in (2, 4, 6):
            yield (
                "clustered_stages",
                {"n_qubits": n_qubits, "repeats": repeats},
                _cluster_stage_circuit(n_qubits=n_qubits, repeats=repeats),
            )

    basis = ("h", "x", "y", "z", "s", "t", "cx", "cz", "swap", "ccx")
    for n_qubits in (8, 10, 12):
        for depth in (60, 90, 120):
            for seed in range(20):
                yield (
                    "seeded_random",
                    {"n_qubits": n_qubits, "depth": depth, "seed": seed},
                    _random_supported_qiskit_circuit(
                        n_qubits=n_qubits,
                        depth=depth,
                        seed=seed,
                        basis_gates=basis,
                    ),
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--init-seg-size", type=int, default=10)
    parser.add_argument("--max-hedge-dist", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
    logging.getLogger("qiskit.compiler").setLevel(logging.WARNING)
    logging.getLogger("hypergraph_partitioner.partitioner").setLevel(logging.DEBUG)

    for idx, (kind, metadata, qiskit_circuit) in enumerate(_candidate_circuits()):
        if idx >= args.max_candidates:
            break

        logger.info("trying candidate=%d kind=%s metadata=%s", idx, kind, metadata)
        circuit = CircuitConverters.from_qiskit(qiskit_circuit)
        result = partition_circuit(
            circuit,
            k=args.k,
            init_seg_size=args.init_seg_size,
            max_hedge_dist=args.max_hedge_dist,
            config_path=KAHYPAR_CONFIG,
        )

        logger.info(
            "candidate=%d result segments=%d boundaries=%d nonlocal=%d teleports=%d",
            idx,
            len(result.segments),
            len(result.boundaries),
            count_nonlocal_interactions(result),
            count_teleports(result),
        )

        if len(result.segments) >= 2:
            print("FOUND")
            print({"kind": kind, **metadata})
            print(f"segments={len(result.segments)} boundaries={len(result.boundaries)}")
            print(f"nonlocal={count_nonlocal_interactions(result)} teleports={count_teleports(result)}")
            return 0

    print("No candidate with >= 2 final segments was found in this deterministic search window.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
