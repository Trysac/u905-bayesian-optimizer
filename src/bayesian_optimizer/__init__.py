"""Bayesian optimization of expensive black-box functions using a Gaussian Process."""

from __future__ import annotations

from bayesian_optimizer import benchmarks
from bayesian_optimizer.acquisition import expected_improvement, lower_confidence_bound
from bayesian_optimizer.errors import ErrorCode, OptimizerError
from bayesian_optimizer.gp import GaussianProcess
from bayesian_optimizer.optimizer import Iteration, Result, minimize, random_search

__all__ = [
    "ErrorCode",
    "GaussianProcess",
    "Iteration",
    "OptimizerError",
    "Result",
    "__version__",
    "benchmarks",
    "expected_improvement",
    "lower_confidence_bound",
    "minimize",
    "random_search",
]

__version__ = "0.1.0"
