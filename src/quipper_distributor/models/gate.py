"""Pydantic v2 discriminated union for Quipper gate variants."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# Wire is an int; negative values are used for ebit ancillae
Wire = int


class SignedWire(BaseModel):
    wire: Wire
    positive: bool  # True = control on |1>, False = control on |0>


# ---------------------------------------------------------------------------
# Gate subtypes — all carry  kind: Literal["<name>"] = "<name>"
# ---------------------------------------------------------------------------


class QGate(BaseModel):
    kind: Literal["QGate"] = "QGate"
    name: str
    inverted: bool = False
    inputs: list[Wire] = Field(default_factory=list)
    outputs: list[Wire] = Field(default_factory=list)
    controls: list[SignedWire] = Field(default_factory=list)
    ncf: bool = False  # no-control flag


class QRot(BaseModel):
    kind: Literal["QRot"] = "QRot"
    name: str
    params: list[float] = Field(default_factory=list)
    inverted: bool = False
    inputs: list[Wire] = Field(default_factory=list)
    outputs: list[Wire] = Field(default_factory=list)
    controls: list[SignedWire] = Field(default_factory=list)
    ncf: bool = False


class QInit(BaseModel):
    kind: Literal["QInit"] = "QInit"
    value: bool  # True = QInit1, False = QInit0
    wire: Wire
    ncf: bool = False


class QTerm(BaseModel):
    kind: Literal["QTerm"] = "QTerm"
    value: bool  # True = QTerm1, False = QTerm0
    wire: Wire
    ncf: bool = False


class QMeas(BaseModel):
    kind: Literal["QMeas"] = "QMeas"
    wire: Wire


class QDiscard(BaseModel):
    kind: Literal["QDiscard"] = "QDiscard"
    wire: Wire


class QPrep(BaseModel):
    kind: Literal["QPrep"] = "QPrep"
    wire: Wire
    ncf: bool = False


class QUnprep(BaseModel):
    kind: Literal["QUnprep"] = "QUnprep"
    wire: Wire
    ncf: bool = False


class CInit(BaseModel):
    kind: Literal["CInit"] = "CInit"
    value: bool
    wire: Wire
    ncf: bool = False


class CTerm(BaseModel):
    kind: Literal["CTerm"] = "CTerm"
    wires: list[Wire] = Field(default_factory=list)
    output: Wire = 0
    ncf: bool = False


class CDiscard(BaseModel):
    kind: Literal["CDiscard"] = "CDiscard"
    wire: Wire


class CNot(BaseModel):
    """Classical not (NOT the usual quantum CNOT gate)."""

    kind: Literal["CNot"] = "CNot"
    inverted: bool = False
    wire: Wire = 0
    target: Wire = 0
    ncf: bool = False


class CGate(BaseModel):
    kind: Literal["CGate"] = "CGate"
    name: str
    inverted: bool = False
    inputs: list[Wire] = Field(default_factory=list)
    output: Wire = 0
    ncf: bool = False


class CGateInv(BaseModel):
    kind: Literal["CGateInv"] = "CGateInv"
    name: str
    inverted: bool = False
    inputs: list[Wire] = Field(default_factory=list)
    output: Wire = 0
    ncf: bool = False


class Comment(BaseModel):
    kind: Literal["Comment"] = "Comment"
    text: str
    inverted: bool = False
    wire_labels: list[tuple[Wire, str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

Gate = Annotated[
    Union[
        QGate,
        QRot,
        QInit,
        QTerm,
        QMeas,
        QDiscard,
        QPrep,
        QUnprep,
        CInit,
        CTerm,
        CDiscard,
        CNot,
        CGate,
        CGateInv,
        Comment,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Helper predicates (matching Common.hs)
# ---------------------------------------------------------------------------


def is_cz(g: object) -> bool:
    """True iff the gate is a QGate with name 'CZ'."""
    return isinstance(g, QGate) and g.name == "CZ"


def is_classical(g: object) -> bool:
    """True for purely classical gate types."""
    return isinstance(g, (CNot, CGate, CGateInv, CInit, CTerm, CDiscard))


def target_of(g: object) -> Wire | None:
    """Return the single-qubit target wire for single-qubit gates, else None."""
    if isinstance(g, QGate):
        if g.name in ("X", "Y", "Z", "S", "T", "H") and len(g.inputs) == 1 and not g.controls:
            return g.inputs[0]
        return None
    if isinstance(g, (QPrep, QUnprep)):
        return g.wire
    if isinstance(g, (QInit, CInit)):
        return g.wire
    if isinstance(g, (QTerm, CTerm)):
        if isinstance(g, QTerm):
            return g.wire
        return None
    if isinstance(g, (QMeas, QDiscard, CDiscard)):
        return g.wire
    return None


def get_wires(g: object) -> list[Wire]:
    """Return all wires of a CZ gate (target + controls)."""
    if isinstance(g, QGate) and g.name == "CZ":
        return g.inputs + [sw.wire for sw in g.controls]
    return []
