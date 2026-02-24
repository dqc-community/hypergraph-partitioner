"""Segment model: Seam variants and Segment container."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from quipper_distributor.models.gate import Wire
from quipper_distributor.models.hypergraph import Hedge


class SeamCompute(BaseModel):
    kind: Literal["compute"] = "compute"


class SeamValue(BaseModel):
    kind: Literal["value"] = "value"
    value: Fraction

    model_config = {"arbitrary_types_allowed": True}


class SeamStop(BaseModel):
    kind: Literal["stop"] = "stop"


Seam = Annotated[
    Union[SeamCompute, SeamValue, SeamStop],
    Field(discriminator="kind"),
]


class Segment(BaseModel):
    # Transitional: supports legacy Gate objects and bosonic_model instructions.
    gates: list[object] = Field(default_factory=list)
    hypergraph: dict[Wire, list[Hedge]] = Field(default_factory=dict)
    partition: dict[Wire, int] = Field(default_factory=dict)
    seam: Seam = Field(default_factory=SeamCompute)
    wire_range: tuple[int, int] = (0, 0)  # (n_qubits, n_wires)

    model_config = {"arbitrary_types_allowed": True}
