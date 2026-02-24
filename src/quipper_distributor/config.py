"""Package-level configuration constants."""

from __future__ import annotations

import os
from pathlib import Path

# KaHyPar imbalance parameter (epsilon)
EPSILON = 0.03

# Default path to KaHyPar configuration file
_PKG_DIR = Path(__file__).parent.parent.parent  # project root
DEFAULT_CONFIG_PATH = str(_PKG_DIR / "kahypar" / "config" / "km1_kKaHyPar_sea20.ini")

# Allow override via environment variable
KAHYPAR_CONFIG = os.environ.get("KAHYPAR_CONFIG", DEFAULT_CONFIG_PATH)
