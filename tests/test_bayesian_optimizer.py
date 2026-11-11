"""Behavior tests for the public API of bayesian_optimizer."""

from __future__ import annotations

import pytest

import bayesian_optimizer


def test_normalize_trims_and_lowercases() -> None:
    assert bayesian_optimizer.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        bayesian_optimizer.normalize(value)


def test_version_is_semver() -> None:
    parts = bayesian_optimizer.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
