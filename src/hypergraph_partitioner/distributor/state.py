"""State containers for annotated-circuit distribution."""

from __future__ import annotations

from dataclasses import dataclass, field

from bosonic_model import Circuit

from hypergraph_partitioner.qpu_utils import QpuLayout


@dataclass(frozen=True)
class PhysicalLocation:
    node: int
    qubit: int


@dataclass
class DistributionState:
    qpu_layouts: dict[int, QpuLayout]
    circuits: dict[int, Circuit]
    qubit_locations: dict[int, PhysicalLocation]
    instruction_index: dict[int, int] = field(default_factory=dict)
    next_order: int = 0
    next_cbit: int = 0
