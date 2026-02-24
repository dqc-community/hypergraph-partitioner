"""Hypergraph, Partition, and Matching models."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, Field

Wire: TypeAlias = int


class Hedge(BaseModel):
    """A hyperedge.

    nan: start marker for span splitting.
    wires: list of (wire_index, position_in_circuit) pairs.
    out_pos: position after which the hedge ends.
    """

    nan: int = 0
    wires: list[tuple[Wire, int]] = Field(default_factory=list)
    out_pos: int = 0


# wire -> list of hedges it participates in
Hypergraph: TypeAlias = dict[Wire, list[Hedge]]

# wire -> QPU block index
Partition: TypeAlias = dict[Wire, int]

# block -> block rename map
Matching: TypeAlias = dict[int, int]
