"""Unit tests for Quipper ASCII parser."""

from __future__ import annotations

from pathlib import Path

from quipper_distributor.models.circuit import WireType
from quipper_distributor.models.gate import (
    Comment,
    QGate,
    QInit,
    QMeas,
    QRot,
    QTerm,
)
from quipper_distributor.parsing.quipper_ascii import emit_circuit, parse_circuit

CIRCUITS_DIR = Path(__file__).parent.parent / "fixtures" / "quipper_circuits"


class TestQGateParsing:
    def test_h_gate(self):
        text = 'Inputs: 0:Qbit\nQGate["H"](0)\nOutputs: 0:Qbit'
        c = parse_circuit(text)
        assert len(c.gates) == 1
        g = c.gates[0]
        assert isinstance(g, QGate)
        assert g.name == "H"
        assert g.inputs == [0]
        assert g.controls == []

    def test_cnot_with_control(self):
        text = 'Inputs: 0:Qbit, 1:Qbit\nQGate["not"](0) with controls=[+1]\nOutputs: 0:Qbit, 1:Qbit'
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, QGate)
        assert g.name == "not"
        assert g.controls[0].wire == 1
        assert g.controls[0].positive is True

    def test_negative_control(self):
        text = 'Inputs: 0:Qbit, 1:Qbit\nQGate["not"](0) with controls=[-1]\nOutputs: 0:Qbit, 1:Qbit'
        c = parse_circuit(text)
        g = c.gates[0]
        assert g.controls[0].positive is False

    def test_cz_gate(self):
        text = 'Inputs: 0:Qbit, 1:Qbit\nQGate["CZ"](0) with controls=[+1]\nOutputs: 0:Qbit, 1:Qbit'
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, QGate)
        assert g.name == "CZ"


class TestQRotParsing:
    def test_qrot_with_control(self):
        text = 'Inputs: 0:Qbit, 1:Qbit\nQRot["R(2pi/%)",4.0](19) with controls=[+18]\nOutputs: 0:Qbit, 1:Qbit'
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, QRot)
        assert g.name == "R(2pi/%)"
        assert g.params == [4.0]

    def test_qrot_no_control(self):
        text = 'Inputs: 0:Qbit\nQRot["Rz",1.5](0)\nOutputs: 0:Qbit'
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, QRot)
        assert g.params == [1.5]
        assert g.controls == []


class TestInitTermParsing:
    def test_qinit0(self):
        text = "Inputs: 0:Qbit\nQInit0(5)\nOutputs: 0:Qbit"
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, QInit)
        assert g.value is False
        assert g.wire == 5

    def test_qinit1(self):
        text = "Inputs: 0:Qbit\nQInit1(3)\nOutputs: 0:Qbit"
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, QInit)
        assert g.value is True

    def test_qterm(self):
        text = "Inputs: 0:Qbit\nQTerm0(0)\nOutputs: 0:Qbit"
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, QTerm)
        assert g.value is False

    def test_measure(self):
        text = "Inputs: 0:Qbit\nMeasure(2)\nOutputs: 0:Qbit"
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, QMeas)
        assert g.wire == 2


class TestCommentParsing:
    def test_comment_basic(self):
        text = 'Inputs: 0:Qbit\nComment["hello"](0:"world")\nOutputs: 0:Qbit'
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, Comment)
        assert g.text == "hello"
        assert g.wire_labels == [(0, "world")]

    def test_comment_multi_labels(self):
        text = 'Inputs: 0:Qbit\nComment["QPU_allocation"](0:"0 QPU", 1:"1 QPU")\nOutputs: 0:Qbit'
        c = parse_circuit(text)
        g = c.gates[0]
        assert isinstance(g, Comment)
        assert len(g.wire_labels) == 2


class TestCircuitParsing:
    def test_simple_circuit(self):
        text = (CIRCUITS_DIR / "simple").read_text()
        c = parse_circuit(text)
        assert len(c.inputs) == 4
        for d in c.inputs:
            assert d.wire_type == WireType.Qbit
        # 3 CNOT gates
        assert len(c.gates) == 3
        for g in c.gates:
            assert isinstance(g, QGate)
            assert g.name == "not"

    def test_qft20_header(self):
        text = (CIRCUITS_DIR / "qft20").read_text()
        c = parse_circuit(text)
        assert len(c.inputs) == 20
        # First few gates should be Comment, QGate["H"], QRot
        gate_kinds = [type(g).__name__ for g in c.gates[:3]]
        assert "QGate" in gate_kinds or "Comment" in gate_kinds


class TestEmitCircuit:
    def test_roundtrip_simple(self):
        text = 'Inputs: 0:Qbit, 1:Qbit\nQGate["H"](0)\nQGate["CZ"](0) with controls=[+1]\nOutputs: 0:Qbit, 1:Qbit'
        c = parse_circuit(text)
        emitted = emit_circuit(c)
        c2 = parse_circuit(emitted)
        assert len(c2.gates) == len(c.gates)
        for g1, g2 in zip(c.gates, c2.gates):
            assert type(g1) is type(g2)

    def test_emit_inputs_outputs(self):
        text = "Inputs: 0:Qbit, 1:Cbit\nOutputs: 0:Qbit, 1:Cbit"
        c = parse_circuit(text)
        emitted = emit_circuit(c)
        assert "0:Qbit" in emitted
        assert "1:Cbit" in emitted
