# hypergraph-partitioner

`hypergraph-partitioner` takes a monolithic circuit, splits it into temporally optimized 'segments', and then used KaHyPar to distribute the gates in each segment to a number of nodes with a fixed number of qubits per node.

Cross node interactions are handled by remote CZ gates (telegate protocol), and cross segment interactions are handled by qubit teleportations (teledata protocol).

The repo is a Python modernization Pablo Andres-Martinez's landmark 2019 paper on hypergraph partitioning applied to DQC: https://arxiv.org/abs/1811.10972

## Example

```python
from bosonic_model.qasm import Translator

from hypergraph_partitioner import (
    annotated_to_distributed_circuit,
    count_interactions,
    count_nonlocal_interactions,
    count_teleports,
    lower_distributed_circuit,
    partition_circuit,
)
from hypergraph_partitioner.config import KAHYPAR_CONFIG

qasm_text = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
cz q[0], q[1];
cz q[2], q[3];
cz q[0], q[3];
"""

circuit = Translator().from_qasm(qasm_text)

result = partition_circuit(
    circuit,
    k=2,
    init_seg_size=10,
    max_hedge_dist=100,
    config_path=KAHYPAR_CONFIG,
)

distributed_symbolic = annotated_to_distributed_circuit(
    result,
    qpu_data_capacity=4,
)

lowered = lower_distributed_circuit(distributed_symbolic)
```


### Telegate Lowering

We decompose (lower) a remote `CZ` through the following telegate protocol:

- Bell-pair primitive
- local `u` / `rzz` gates
- LOCC corrections


![Telegate lowering](.pytest_artifacts/remote_cz_protocol.png)

### Teledata Lowering

We lower a qubit teleportation protocol (teledata) through the following protocol:

- Bell-pair primitive
- local `u` / `rzz` gates
- LOCC corrections on the destination side


![Teledata lowering](.pytest_artifacts/teledata_protocol.png)

Useful examples:

- `examples/basic_partition_stats.py`
- `examples/contrived_two_phase_four_qubit_segments.py`

Tests:
```
make test
```