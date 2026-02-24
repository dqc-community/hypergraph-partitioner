"""Unit tests for Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from quipper_distributor.models.gate import (
    CDiscard,
    CGate,
    CGateInv,
    CInit,
    CNot,
    Comment,
    CTerm,
    Gate,
    QDiscard,
    QGate,
    QInit,
    QMeas,
    QPrep,
    QRot,
    QTerm,
    QUnprep,
    SignedWire,
    get_wires,
    is_classical,
    is_cz,
    target_of,
)
from quipper_distributor.models.circuit import Circuit, WireDecl, WireType
from quipper_distributor.models.hypergraph import Hedge, Hypergraph
from quipper_distributor.models.segment import Segment, SeamCompute, SeamStop, SeamValue

gate_adapter = TypeAdapter(Gate)


class TestQGate:
    def test_basic(self):
        g = QGate(name="H", inputs=[0])
        assert g.kind == "QGate"
        assert g.name == "H"
        assert g.inputs == [0]
        assert not g.inverted

    def test_with_controls(self):
        g = QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])
        assert g.controls[0].wire == 1
        assert g.controls[0].positive is True

    def test_discriminator(self):
        data = {"kind": "QGate", "name": "X", "inputs": [2]}
        g = gate_adapter.validate_python(data)
        assert isinstance(g, QGate)

    def test_serialisation(self):
        g = QGate(name="T", inputs=[3], inverted=True)
        d = g.model_dump()
        assert d["kind"] == "QGate"
        assert d["inverted"] is True


class TestQRot:
    def test_basic(self):
        g = QRot(
            name="R(2pi/%)",
            params=[4.0],
            inputs=[19],
            controls=[SignedWire(wire=18, positive=True)],
        )
        assert g.kind == "QRot"
        assert g.params == [4.0]
        assert g.inputs == [19]

    def test_discriminator(self):
        data = {"kind": "QRot", "name": "Rz", "params": [1.5], "inputs": [0]}
        g = gate_adapter.validate_python(data)
        assert isinstance(g, QRot)


class TestQInit:
    def test_init0(self):
        g = QInit(value=False, wire=5)
        assert g.value is False
        assert g.wire == 5
        assert gate_adapter.validate_python(g.model_dump()).kind == "QInit"

    def test_init1(self):
        g = QInit(value=True, wire=3)
        assert g.value is True


class TestQTerm:
    def test_term(self):
        g = QTerm(value=True, wire=2)
        assert isinstance(gate_adapter.validate_python(g.model_dump()), QTerm)


class TestOtherGates:
    def test_qmeas(self):
        g = QMeas(wire=7)
        assert g.kind == "QMeas"

    def test_qdiscard(self):
        g = QDiscard(wire=0)
        assert isinstance(gate_adapter.validate_python(g.model_dump()), QDiscard)

    def test_qprep(self):
        g = QPrep(wire=1)
        assert g.kind == "QPrep"

    def test_qunprep(self):
        g = QUnprep(wire=2)
        assert isinstance(gate_adapter.validate_python(g.model_dump()), QUnprep)

    def test_cinit(self):
        g = CInit(value=False, wire=3)
        assert g.kind == "CInit"

    def test_cterm(self):
        g = CTerm(wires=[4], output=4)
        assert g.kind == "CTerm"

    def test_cdiscard(self):
        g = CDiscard(wire=5)
        assert isinstance(gate_adapter.validate_python(g.model_dump()), CDiscard)

    def test_cnot(self):
        g = CNot(wire=0, target=1)
        assert g.kind == "CNot"

    def test_cgate(self):
        g = CGate(name="and", inputs=[0, 1], output=2)
        assert isinstance(gate_adapter.validate_python(g.model_dump()), CGate)

    def test_cgateinv(self):
        g = CGateInv(name="and", inputs=[0, 1], output=2)
        assert isinstance(gate_adapter.validate_python(g.model_dump()), CGateInv)

    def test_comment(self):
        g = Comment(text="test", wire_labels=[(0, "a"), (1, "b")])
        assert g.text == "test"
        assert g.wire_labels == [(0, "a"), (1, "b")]


class TestHelperPredicates:
    def test_is_cz_true(self):
        g = QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])
        assert is_cz(g)

    def test_is_cz_false(self):
        assert not is_cz(QGate(name="H", inputs=[0]))
        assert not is_cz(QMeas(wire=0))

    def test_is_classical(self):
        assert is_classical(CNot(wire=0, target=1))
        assert is_classical(CGate(name="and", inputs=[0], output=1))
        assert is_classical(CDiscard(wire=0))
        assert not is_classical(QGate(name="H", inputs=[0]))
        assert not is_classical(QMeas(wire=0))

    def test_target_of_single_qubit(self):
        for name in ("X", "Y", "Z", "S", "T", "H"):
            g = QGate(name=name, inputs=[3])
            assert target_of(g) == 3

    def test_target_of_multi_qubit(self):
        g = QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])
        assert target_of(g) is None

    def test_target_of_qmeas(self):
        assert target_of(QMeas(wire=5)) == 5

    def test_get_wires_cz(self):
        g = QGate(name="CZ", inputs=[0], controls=[SignedWire(wire=1, positive=True)])
        assert get_wires(g) == [0, 1]

    def test_get_wires_non_cz(self):
        assert get_wires(QGate(name="H", inputs=[0])) == []


class TestCircuitModel:
    def test_wire_decl(self):
        wd = WireDecl(wire=0, wire_type=WireType.Qbit)
        assert wd.wire_type == WireType.Qbit

    def test_circuit(self):
        c = Circuit(
            inputs=[WireDecl(wire=0, wire_type=WireType.Qbit)],
            outputs=[WireDecl(wire=0, wire_type=WireType.Qbit)],
            gates=[QGate(name="H", inputs=[0])],
        )
        assert len(c.gates) == 1


class TestHypergraphModel:
    def test_hedge(self):
        h = Hedge(nan=0, wires=[(1, 5)], out_pos=6)
        assert h.wires == [(1, 5)]

    def test_empty_hypergraph(self):
        hyp: Hypergraph = {}
        assert len(hyp) == 0


class TestSegmentModel:
    def test_seam_compute(self):
        s = SeamCompute()
        assert s.kind == "compute"

    def test_seam_value(self):
        from fractions import Fraction

        s = SeamValue(value=Fraction(1, 3))
        assert s.value == Fraction(1, 3)

    def test_seam_stop(self):
        s = SeamStop()
        assert s.kind == "stop"

    def test_segment(self):
        seg = Segment(
            gates=[QGate(name="H", inputs=[0])],
            seam=SeamCompute(),
            wire_range=(0, 0),
        )
        assert len(seg.gates) == 1
