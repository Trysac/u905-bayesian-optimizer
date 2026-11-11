"""Custom exception classes for bayesian_optimizer."""

from __future__ import annotations

from typing import Literal

ErrorCode = Literal[
    "invalid_bounds",
    "invalid_budget",
    "non_finite_value",
    "invalid_acquisition",
    "not_fitted",
    "numerical_error",
]


class OptimizerError(Exception):
    """Base exception raised by bayesian_optimizer with a machine-readable code."""

    code: ErrorCode

    def __init__(self, code: ErrorCode, message: str) -> None:
        clipped = message[:200]
        super().__init__(clipped)
        self.code = code
