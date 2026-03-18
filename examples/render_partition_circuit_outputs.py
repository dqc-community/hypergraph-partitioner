"""Render the original, symbolic, and lowered partition-circuit outputs.

This example uses the public ``partition_circuit(...)`` API in both symbolic
and lowered modes, then renders:

- the original monolithic input circuit
- the symbolic distributed circuit
- the lowered distributed circuit

All three circuit diagrams are saved as PNGs using Qiskit's mpl drawer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from bosonic_converters import CircuitConverters
from bosonic_model.qasm import Translator

from hypergraph_partitioner import partition_circuit

matplotlib.use("Agg")


QASM_TEXT = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    y q[3];
    swap q[0],q[2];
    swap q[1],q[3];
    x q[3];
    swap q[3],q[0];
    cz q[3],q[0];
    swap q[0],q[1];
    x q[1];
    """


def _save_circuit_image(path: Path, *, title: str, circuit) -> None:
    figure = circuit.draw(output="mpl", idle_wires=False, fold=-1)
    figure.suptitle(title)
    figure.savefig(path, bbox_inches="tight")
    figure.clf()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".pytest_artifacts" / "example_partition_circuit",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    circuit = Translator().from_qasm(QASM_TEXT)
    symbolic = partition_circuit(
        circuit,
        nodes=2,
        qubits_per_node=2,
        init_seg_size=2,
        max_hedge_dist=100,
        output="symbolic",
    )
    lowered = partition_circuit(
        circuit,
        nodes=2,
        qubits_per_node=2,
        init_seg_size=2,
        max_hedge_dist=100,
        output="lowered",
    )

    input_qiskit = CircuitConverters.to_qiskit(circuit)
    symbolic_qiskit = CircuitConverters.to_qiskit(symbolic.as_monolithic_circuit())
    lowered_qiskit = CircuitConverters.to_qiskit(lowered.as_monolithic_circuit())

    input_path = output_dir / "input_monolithic.png"
    symbolic_path = output_dir / "symbolic_distributed.png"
    lowered_path = output_dir / "lowered_distributed.png"

    _save_circuit_image(input_path, title="Input Monolithic Circuit", circuit=input_qiskit)
    _save_circuit_image(
        symbolic_path,
        title="Symbolic Distributed Circuit",
        circuit=symbolic_qiskit,
    )
    _save_circuit_image(
        lowered_path,
        title="Lowered Distributed Circuit",
        circuit=lowered_qiskit,
    )

    print("Wrote circuit diagrams:")
    print(f"  {input_path}")
    print(f"  {symbolic_path}")
    print(f"  {lowered_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
