"""Unit tests for core hypergraph/segment models."""

from __future__ import annotations

from fractions import Fraction

from hypergraph_partitioner.models.hypergraph import Hedge
from hypergraph_partitioner.models.segment import SeamCompute, SeamStop, SeamValue, Segment


def test_hedge_model_roundtrip() -> None:
    hedge = Hedge(nan=0, wires=[(1, 5)], out_pos=6)

    dumped = hedge.model_dump()

    assert dumped["wires"] == [(1, 5)]
    assert dumped["out_pos"] == 6


def test_segment_defaults() -> None:
    seg = Segment()

    assert seg.gates == []
    assert seg.hypergraph == {}
    assert seg.partition == {}
    assert isinstance(seg.seam, SeamCompute)


def test_segment_with_explicit_seams() -> None:
    seg_value = Segment(seam=SeamValue(value=Fraction(1, 3)))
    seg_stop = Segment(seam=SeamStop())

    assert isinstance(seg_value.seam, SeamValue)
    assert seg_value.seam.value == Fraction(1, 3)
    assert isinstance(seg_stop.seam, SeamStop)
