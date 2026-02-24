"""Circuit model: Wire declarations and Circuit container."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from quipper_distributor.models.gate import Gate, Wire


class WireType(str, Enum):
    Qbit = "Qbit"
    Cbit = "Cbit"


class WireDecl(BaseModel):
    wire: Wire
    wire_type: WireType


class Circuit(BaseModel):
    inputs: list[WireDecl] = Field(default_factory=list)
    outputs: list[WireDecl] = Field(default_factory=list)
    gates: list[Gate] = Field(default_factory=list)
