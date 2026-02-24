"""Shared circuit fixtures for tests."""

from __future__ import annotations

SIMPLE_CIRCUIT = """\
Inputs: 0:Qbit, 1:Qbit, 2:Qbit, 3:Qbit
QGate["not"](3) with controls=[+1]
QGate["not"](2) with controls=[+1]
QGate["not"](0) with controls=[+1]
Outputs: 0:Qbit, 1:Qbit, 2:Qbit, 3:Qbit
"""

MINIMAL_CZ = """\
Inputs: 0:Qbit, 1:Qbit
QGate["CZ"](0) with controls=[+1]
Outputs: 0:Qbit, 1:Qbit
"""

QFT_FIRST_20_LINES = """\
Inputs: 0:Qbit, 1:Qbit, 2:Qbit, 3:Qbit, 4:Qbit, 5:Qbit, 6:Qbit, 7:Qbit, 8:Qbit, 9:Qbit, 10:Qbit, 11:Qbit, 12:Qbit, 13:Qbit, 14:Qbit, 15:Qbit, 16:Qbit, 17:Qbit, 18:Qbit, 19:Qbit
Comment["ENTER: qft_rev"](0:"qs[0]", 1:"qs[1]")
QGate["H"](19)
QRot["R(2pi/%)",4.0](19) with controls=[+18]
QGate["H"](18)
Outputs: 0:Qbit, 1:Qbit, 2:Qbit, 3:Qbit, 4:Qbit, 5:Qbit, 6:Qbit, 7:Qbit, 8:Qbit, 9:Qbit, 10:Qbit, 11:Qbit, 12:Qbit, 13:Qbit, 14:Qbit, 15:Qbit, 16:Qbit, 17:Qbit, 18:Qbit, 19:Qbit
"""

WITH_TOFFOLIS = """\
Inputs: 0:Qbit, 1:Qbit, 2:Qbit
QGate["not"](0) with controls=[+1, +2]
Outputs: 0:Qbit, 1:Qbit, 2:Qbit
"""

WITH_SWAP = """\
Inputs: 0:Qbit, 1:Qbit
QGate["swap"](0, 1)
Outputs: 0:Qbit, 1:Qbit
"""
