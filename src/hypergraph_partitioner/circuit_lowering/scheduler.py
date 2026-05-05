"""Resource scheduler for lowered remote operations."""

from __future__ import annotations

from dataclasses import dataclass, field

from hypergraph_partitioner.qpu_utils import QpuLayout, alloc_comm, free_comm


class RemoteScheduleError(RuntimeError):
    """Raised when remote operations would overlap on the same node."""


@dataclass(frozen=True)
class CommLease:
    node: int
    qubit: int
    role: str


@dataclass
class RemoteOperation:
    scheduler: RemoteOperationScheduler
    nodes: tuple[int, ...]
    leases: list[CommLease] = field(default_factory=list)

    def alloc_comm(self, node: int, role: str) -> int:
        lease = self.scheduler.alloc_comm(node, role)
        self.leases.append(lease)
        return lease.qubit

    def release_all(self) -> None:
        while self.leases:
            self.scheduler.release_comm(self.leases.pop())


class RemoteOperationScheduler:
    """Serial resource allocator for remote lowering protocols.

    The current lowering pass emits one remote protocol at a time. This scheduler
    makes that contract explicit and centralizes comm-qubit allocation so later
    boundary-swap lowering can share the same resource policy.
    """

    def __init__(self, qpu_layouts: dict[int, QpuLayout]) -> None:
        self.qpu_layouts = qpu_layouts
        self._active_nodes: set[int] = set()
        self._peak_comm_usage: dict[int, int] = {}

    def begin(self, nodes: tuple[int, ...]) -> RemoteOperation:
        unique_nodes = tuple(dict.fromkeys(nodes))
        overlap = self._active_nodes.intersection(unique_nodes)
        if overlap:
            raise RemoteScheduleError(
                f"remote operation overlaps active node(s): {sorted(overlap)}"
            )
        self._active_nodes.update(unique_nodes)
        return RemoteOperation(scheduler=self, nodes=unique_nodes)

    def finish(self, operation: RemoteOperation) -> None:
        operation.release_all()
        for node in operation.nodes:
            self._active_nodes.remove(node)

    def alloc_comm(self, node: int, role: str) -> CommLease:
        layout = self.qpu_layouts[node]
        qubit = alloc_comm(layout)
        in_use = len(layout.comm_slots) - len(layout.free_comm)
        self._peak_comm_usage[node] = max(self._peak_comm_usage.get(node, 0), in_use)
        return CommLease(node=node, qubit=qubit, role=role)

    def release_comm(self, lease: CommLease) -> None:
        free_comm(self.qpu_layouts[lease.node], lease.qubit)

    def peak_comm_usage(self, node: int) -> int:
        return self._peak_comm_usage.get(node, 0)
