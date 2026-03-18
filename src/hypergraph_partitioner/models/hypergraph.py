"""Explicit hypergraph model used by partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

Partition: TypeAlias = dict[int, int]
Matching: TypeAlias = dict[int, int]


@dataclass(frozen=True)
class QubitVertex:
    qubit_id: int


@dataclass(frozen=True)
class InteractionVertex:
    interaction_id: int
    position: int
    qubits: tuple[int, ...]


@dataclass
class Hypergraph:
    qubits: dict[int, QubitVertex]
    interactions: dict[int, InteractionVertex]

    @cached_property
    def qubit_to_interactions(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {qubit_id: [] for qubit_id in self.qubits}

        for interaction_id, interaction in self.interactions.items():
            for qubit_id in interaction.qubits:
                result.setdefault(qubit_id, []).append(interaction_id)

        for interaction_ids in result.values():
            interaction_ids.sort(key=lambda i: self.interactions[i].position)

        return result
