"""Unit tests for remote-operation resource scheduling."""

from __future__ import annotations

import pytest

from hypergraph_partitioner.circuit_lowering.scheduler import (
    RemoteOperationScheduler,
    RemoteScheduleError,
)
from hypergraph_partitioner.qpu_utils import build_qpu_layouts


def test_scheduler_reuses_comm_qubits_after_operation_finish() -> None:
    layouts = build_qpu_layouts(qubits_per_node=1, n_nodes=2)
    scheduler = RemoteOperationScheduler(layouts)

    first = scheduler.begin((0, 1))
    q0 = first.alloc_comm(0, "first.source")
    q1 = first.alloc_comm(1, "first.target")
    scheduler.finish(first)

    second = scheduler.begin((0, 1))
    assert second.alloc_comm(0, "second.source") == q0
    assert second.alloc_comm(1, "second.target") == q1
    scheduler.finish(second)

    assert scheduler.peak_comm_usage(0) == 1
    assert scheduler.peak_comm_usage(1) == 1


def test_scheduler_rejects_overlapping_operations_on_same_node() -> None:
    layouts = build_qpu_layouts(qubits_per_node=1, n_nodes=3)
    scheduler = RemoteOperationScheduler(layouts)

    active = scheduler.begin((0, 1))
    with pytest.raises(RemoteScheduleError, match="active node"):
        scheduler.begin((1, 2))

    scheduler.finish(active)
