"""Explicit hypergraph model used by partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

Wire: TypeAlias = int
Interaction: TypeAlias = int
Partition: TypeAlias = dict[Wire, int]
Matching: TypeAlias = dict[int, int]


@dataclass(frozen=True)
class WireVertex:
    wire_id: int


@dataclass(frozen=True)
class InteractionVertex:
    interaction_id: int
    position: int
    qubits: tuple[int, ...]


@dataclass
class Hypergraph:
    wires: dict[int, WireVertex]
    interactions: dict[int, InteractionVertex]

    @cached_property
    def wire_to_interactions(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {wire_id: [] for wire_id in self.wires}

        for interaction_id, interaction in self.interactions.items():
            for wire_id in interaction.qubits:
                result.setdefault(wire_id, []).append(interaction_id)

        for interaction_ids in result.values():
            interaction_ids.sort(key=lambda i: self.interactions[i].position)

        return result
