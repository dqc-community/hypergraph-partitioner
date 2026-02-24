"""pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require kahypar)"
    )
