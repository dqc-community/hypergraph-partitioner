"""Hypergraph, Partition, and Matching models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from quipper_distributor.models.gate import Wire


class Hedge(BaseModel):
    """A hyperedge.

    nan: unused marker (0), matching Haskell original.
    wires: list of (wire_index, position_in_circuit) pairs.
    out_pos: position after which the hedge ends.
    """

    nan: int = 0
    wires: list[tuple[Wire, int]] = Field(default_factory=list)
    out_pos: int = 0


# wire → list of hedges it "controls"
Hypergraph = dict[Wire, list[Hedge]]

# wire → QPU block index
Partition = dict[Wire, int]

# block → block rename map
Matching = dict[int, int]
