"""Smoke tests for package metadata."""

from __future__ import annotations

import bayesian_optimizer


def test_version() -> None:
    assert bayesian_optimizer.__version__ == "0.1.0"
