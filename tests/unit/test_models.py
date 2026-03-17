"""Unit tests for core hypergraph/segment models."""

from __future__ import annotations

from fractions import Fraction

from bosonic_model.instructions import UInstruction

from hypergraph_partitioner.models.circuit_annotations import (
    NodeId,
    BoundaryId,
    LocalOp,
    PartitionedCircuit,
    PartitionedSegment,
    SegmentBoundary,
    SegmentId,
    TeleportBoundary,
    QubitId,
)
from hypergraph_partitioner.models.hypergraph import Hypergraph, InteractionVertex, QubitVertex
from hypergraph_partitioner.models.segment import SeamCompute, SeamStop, SeamValue, Segment


def test_hypergraph_qubit_to_interactions_indexes_by_wire() -> None:
    hyp = Hypergraph(
        wires={0: QubitVertex(0), 1: QubitVertex(1), 2: QubitVertex(2)},
        interactions={
            10: InteractionVertex(interaction_id=10, position=5, qubits=(0, 1)),
            11: InteractionVertex(interaction_id=11, position=9, qubits=(1, 2)),
        },
    )

    assert hyp.qubit_to_interactions == {0: [10], 1: [10, 11], 2: [11]}


def test_segment_defaults() -> None:
    seg = Segment()

    assert seg.gates == []
    assert seg.hypergraph == Hypergraph(qubits={}, interactions={})
    assert seg.partition == {}
    assert isinstance(seg.seam, SeamCompute)


def test_segment_with_explicit_seams() -> None:
    seg_value = Segment(seam=SeamValue(value=Fraction(1, 3)))
    seg_stop = Segment(seam=SeamStop())

    assert isinstance(seg_value.seam, SeamValue)
    assert seg_value.seam.value == Fraction(1, 3)
    assert isinstance(seg_stop.seam, SeamStop)


def test_partitioned_circuit_model_roundtrip() -> None:
    inst = UInstruction(qubit=0, qubits=[0], theta=0.0, phi=0.0, lam=0.0, params=[0.0, 0.0, 0.0])
    segment = PartitionedSegment(
        segment_id=SegmentId(0),
        instructions=[inst],
        partition={QubitId(0): NodeId(1)},
    )
    teleport = TeleportBoundary(qubit=QubitId(0), from_node=NodeId(0), to_node=NodeId(1))
    boundary = SegmentBoundary(
        boundary_id=BoundaryId(0),
        left_segment_id=SegmentId(0),
        right_segment_id=SegmentId(1),
        teleports=[teleport],
    )
    result = PartitionedCircuit(segments=[segment], boundaries=[boundary])

    assert result.segments[0].partition[QubitId(0)] == NodeId(1)
    assert result.boundaries[0].teleports[0].to_node == NodeId(1)


def test_local_op_blocks_preserve_block_ids() -> None:
    inst = UInstruction(qubit=0, qubits=[0], theta=0.0, phi=0.0, lam=0.0, params=[0.0, 0.0, 0.0])
    op = LocalOp(
        segment_id=SegmentId(0),
        instruction=inst,
        nodes=(NodeId(2),),
    )

    assert op.nodes == (NodeId(2),)
