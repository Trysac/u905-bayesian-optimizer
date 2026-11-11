"""Bayesian optimization and random search minimization loops."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from bayesian_optimizer.acquisition import expected_improvement, lower_confidence_bound
from bayesian_optimizer.errors import OptimizerError
from bayesian_optimizer.gp import GaussianProcess

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

AcquisitionName = Literal["ei", "lcb"]

_MIN_DIMS = 1
_MAX_DIMS = 20
_MAX_CALLS = 500
_N_CANDIDATES = 2048
_N_SEEDS_REFINE = 5
_N_REFINE_STEPS = 20


@dataclass(frozen=True)
class Result:
    """Optimization result returned by ``minimize`` and ``random_search``."""

    x: list[float]
    fun: float
    xs: list[list[float]]
    ys: list[float]


@dataclass(frozen=True)
class Iteration:
    """Progress snapshot passed to ``on_iteration`` after each objective evaluation."""

    index: int
    x: list[float]
    y: float
    best_y: float


def _validate_bounds(bounds: Sequence[tuple[float, float]]) -> npt.NDArray[np.float64]:
    """Validate dimension count and finite low < high bounds."""
    n_dims = len(bounds)
    if n_dims < _MIN_DIMS or n_dims > _MAX_DIMS:
        raise OptimizerError(
            "invalid_bounds",
            f"Number of dimensions must be between {_MIN_DIMS} and {_MAX_DIMS}, got {n_dims}.",
        )
    b_arr = np.empty((n_dims, 2), dtype=np.float64)
    for i, pair in enumerate(bounds):
        if len(pair) != 2:
            raise OptimizerError(
                "invalid_bounds",
                f"Bound at index {i} must be a (low, high) pair.",
            )
        low, high = float(pair[0]), float(pair[1])
        if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
            raise OptimizerError(
                "invalid_bounds",
                f"Bound at index {i} must have finite low < high, got ({low}, {high}).",
            )
        b_arr[i, 0] = low
        b_arr[i, 1] = high
    return b_arr


def _validate_budget(n_calls: int, n_initial: int) -> None:
    """Validate 1 <= n_initial <= n_calls <= 500."""
    if (
        isinstance(n_calls, bool)
        or isinstance(n_initial, bool)
        or n_initial < 1
        or n_calls < n_initial
        or n_calls > _MAX_CALLS
    ):
        raise OptimizerError(
            "invalid_budget",
            f"Budget must satisfy 1 <= n_initial <= n_calls <= {_MAX_CALLS}.",
        )


def _evaluate_objective(
    f: Callable[[list[float]], float],
    point: list[float],
) -> float:
    """Call the user objective on a copy of point and verify finite float return."""
    val = float(f(list(point)))
    if not math.isfinite(val):
        raise OptimizerError(
            "non_finite_value",
            "Objective function returned a non-finite value (NaN or Inf).",
        )
    return val


def _score_desirability(
    gp: GaussianProcess,
    candidates: npt.NDArray[np.float64],
    best_y: float,
    acquisition: AcquisitionName,
    xi: float,
    kappa: float,
) -> npt.NDArray[np.float64]:
    """Compute acquisition desirability score (higher is better for candidate selection)."""
    mu, sigma = gp.predict(candidates)
    if acquisition == "ei":
        return expected_improvement(mu, sigma, best=best_y, xi=xi)
    return -lower_confidence_bound(mu, sigma, kappa=kappa)


def _refine_candidate(
    gp: GaussianProcess,
    start: npt.NDArray[np.float64],
    bounds_arr: npt.NDArray[np.float64],
    best_y: float,
    acquisition: AcquisitionName,
    xi: float,
    kappa: float,
) -> tuple[npt.NDArray[np.float64], float]:
    """Deterministically refine a candidate with 20 shrinking coordinate-wise steps."""
    low = bounds_arr[:, 0]
    high = bounds_arr[:, 1]
    span = high - low
    n_dims = start.shape[0]

    current = start.copy()
    current_score = float(
        _score_desirability(
            gp,
            current[np.newaxis, :],
            best_y,
            acquisition,
            xi,
            kappa,
        )[0]
    )

    for step in range(_N_REFINE_STEPS):
        frac = 0.25 * (0.85**step)
        probes = np.repeat(current[np.newaxis, :], 2 * n_dims, axis=0)
        for d in range(n_dims):
            delta = frac * span[d]
            probes[2 * d, d] = min(high[d], current[d] + delta)
            probes[2 * d + 1, d] = max(low[d], current[d] - delta)
        scores = _score_desirability(gp, probes, best_y, acquisition, xi, kappa)
        best_probe_idx = int(np.argmax(scores))
        best_probe_score = float(scores[best_probe_idx])
        if best_probe_score > current_score:
            current = probes[best_probe_idx].copy()
            current_score = best_probe_score

    return current, current_score


def random_search(
    f: Callable[[list[float]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    n_calls: int,
    seed: int = 0,
) -> Result:
    """Minimize objective ``f`` by uniform random sampling within ``bounds``."""
    bounds_arr = _validate_bounds(bounds)
    _validate_budget(n_calls=n_calls, n_initial=1)

    rng = np.random.default_rng(seed)
    low = bounds_arr[:, 0]
    high = bounds_arr[:, 1]
    n_dims = bounds_arr.shape[0]

    xs: list[list[float]] = []
    ys: list[float] = []
    best_y = math.inf
    best_x: list[float] = []

    for _ in range(n_calls):
        sample = rng.uniform(low, high, size=n_dims)
        point = [float(v) for v in sample]
        y_val = _evaluate_objective(f, point)
        xs.append(point)
        ys.append(y_val)
        if y_val < best_y:
            best_y = y_val
            best_x = list(point)

    return Result(x=best_x, fun=best_y, xs=xs, ys=ys)


def minimize(
    f: Callable[[list[float]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    n_calls: int = 30,
    n_initial: int = 5,
    acquisition: AcquisitionName = "ei",
    xi: float = 0.01,
    kappa: float = 2.0,
    seed: int = 0,
    on_iteration: Callable[[Iteration], None] | None = None,
) -> Result:
    """Minimize black-box function ``f`` over ``bounds`` using Gaussian Process BO."""
    bounds_arr = _validate_bounds(bounds)
    _validate_budget(n_calls=n_calls, n_initial=n_initial)
    if acquisition not in ("ei", "lcb"):
        raise OptimizerError(
            "invalid_acquisition",
            f"Invalid acquisition function {acquisition!r}; expected 'ei' or 'lcb'.",
        )
    if not (math.isfinite(xi) and math.isfinite(kappa)):
        raise OptimizerError(
            "invalid_acquisition",
            "Acquisition hyperparameters xi and kappa must be finite numbers.",
        )

    rng = np.random.default_rng(seed)
    low = bounds_arr[:, 0]
    high = bounds_arr[:, 1]
    n_dims = bounds_arr.shape[0]

    xs: list[list[float]] = []
    ys: list[float] = []
    best_y = math.inf
    best_x: list[float] = []

    for i in range(n_calls):
        if i < n_initial:
            sample = rng.uniform(low, high, size=n_dims)
            point = [float(v) for v in sample]
        else:
            gp = GaussianProcess(kernel="matern52", noise=1e-6)
            gp.fit(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), bounds_arr)

            candidates = rng.uniform(low, high, size=(_N_CANDIDATES, n_dims))
            scores = _score_desirability(gp, candidates, best_y, acquisition, xi, kappa)
            top_indices = np.argsort(scores)[-_N_SEEDS_REFINE:][::-1]

            best_cand: npt.NDArray[np.float64] = candidates[int(top_indices[0])].copy()
            best_cand_score = float(scores[int(top_indices[0])])

            for idx in top_indices:
                refined, ref_score = _refine_candidate(
                    gp,
                    candidates[int(idx)],
                    bounds_arr,
                    best_y,
                    acquisition,
                    xi,
                    kappa,
                )
                if ref_score > best_cand_score:
                    best_cand = refined
                    best_cand_score = ref_score

            point = [float(v) for v in best_cand]

        y_val = _evaluate_objective(f, point)
        xs.append(point)
        ys.append(y_val)
        if y_val < best_y:
            best_y = y_val
            best_x = list(point)

        logger.debug("Iteration %d finished with best_y=%g", i, best_y)
        if on_iteration is not None:
            on_iteration(Iteration(index=i, x=list(point), y=y_val, best_y=best_y))

    return Result(x=best_x, fun=best_y, xs=xs, ys=ys)
