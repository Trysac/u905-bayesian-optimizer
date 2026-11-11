"""Tests for bayesian_optimizer covering all acceptance criteria and edge cases."""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
import pytest

from bayesian_optimizer import (
    GaussianProcess,
    Iteration,
    OptimizerError,
    Result,
    benchmarks,
    expected_improvement,
    lower_confidence_bound,
    minimize,
    random_search,
)
from bayesian_optimizer.benchmarks import branin, hartmann6
from bayesian_optimizer.gp import _cholesky_with_jitter


def test_benchmarks_known_optima() -> None:
    assert branin(math.pi, 2.275) == pytest.approx(0.39788735772973816, abs=1e-9)
    assert branin([math.pi, 2.275]) == pytest.approx(0.39788735772973816, abs=1e-9)
    assert hartmann6([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]) == pytest.approx(
        -3.322368011391339, abs=1e-9
    )
    assert benchmarks.branin(math.pi, 2.275) == pytest.approx(0.39788735772973816, abs=1e-9)


def test_benchmarks_input_validation() -> None:
    with pytest.raises(TypeError):
        branin(1.0)
    with pytest.raises(ValueError):
        branin([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        branin([1.0, 2.0], 3.0)
    with pytest.raises(ValueError):
        hartmann6([0.1, 0.2])


def test_acquisition_functions_exact_values() -> None:
    assert expected_improvement(np.array([0.0]), np.array([1.0]), best=0.0, xi=0.0)[
        0
    ] == pytest.approx(0.3989422804014327, abs=1e-12)
    assert expected_improvement(np.array([1.0]), np.array([1.0]), best=0.0, xi=0.0)[
        0
    ] == pytest.approx(0.08331547058768629, abs=1e-12)
    assert expected_improvement(np.array([0.0]), np.array([0.0]), best=0.0, xi=0.0)[0] == 0.0
    assert lower_confidence_bound(np.array([1.0]), np.array([0.5]), kappa=2.0)[0] == 0.0


def test_gaussian_process_interpolation() -> None:
    gp = GaussianProcess(noise=1e-10).fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    mean, std = gp.predict(np.array([[0.0], [1.0]]))
    assert mean[0] == pytest.approx(0.0, abs=1e-6)
    assert mean[1] == pytest.approx(1.0, abs=1e-6)
    assert float(np.max(std)) < 1e-3


def test_gaussian_process_rbf_and_constant_targets() -> None:
    gp = GaussianProcess(kernel="rbf", noise=1e-8).fit(
        np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
        np.array([5.0, 5.0, 5.0]),
    )
    mean, std = gp.predict(np.array([[0.5, 1.0]]))
    assert mean[0] == pytest.approx(5.0, abs=1e-5)
    assert std[0] >= 0.0


def test_gaussian_process_errors() -> None:
    with pytest.raises(OptimizerError) as exc_info:
        GaussianProcess(kernel="invalid")  # type: ignore[arg-type]
    assert exc_info.value.code == "numerical_error"

    with pytest.raises(OptimizerError) as exc_info:
        GaussianProcess(noise=-1.0)
    assert exc_info.value.code == "numerical_error"

    gp = GaussianProcess()
    with pytest.raises(OptimizerError) as exc_info:
        gp.predict(np.array([[0.0]]))
    assert exc_info.value.code == "not_fitted"

    with pytest.raises(OptimizerError) as exc_info:
        gp.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert exc_info.value.code == "numerical_error"

    with pytest.raises(OptimizerError) as exc_info:
        gp.fit(np.empty((0, 1)), np.empty(0))
    assert exc_info.value.code == "numerical_error"

    with pytest.raises(OptimizerError) as exc_info:
        gp.fit(np.zeros((2, 21)), np.zeros(2))
    assert exc_info.value.code == "invalid_bounds"

    with pytest.raises(OptimizerError) as exc_info:
        gp.fit(np.array([[0.0], [float("nan")]]), np.array([0.0, 1.0]))
    assert exc_info.value.code == "non_finite_value"

    gp.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    with pytest.raises(OptimizerError) as exc_info:
        gp.predict(np.array([[0.0, 1.0]]))
    assert exc_info.value.code == "numerical_error"

    with pytest.raises(OptimizerError) as exc_info:
        gp.predict(np.array([[float("inf")]]))
    assert exc_info.value.code == "non_finite_value"


def test_cholesky_jitter_escalation_and_failure() -> None:
    # Slightly negative eigenvalue triggers jitter escalation and succeeds
    mat = np.array([[1.0, 1.0 + 1e-8], [1.0 + 1e-8, 1.0]], dtype=np.float64)
    chol = _cholesky_with_jitter(mat, noise=0.0)
    assert chol.shape == (2, 2)

    # Severely indefinite matrix fails even at 1e-4 jitter
    bad_mat = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    with pytest.raises(OptimizerError) as exc_info:
        _cholesky_with_jitter(bad_mat, noise=0.0)
    assert exc_info.value.code == "numerical_error"


def test_determinism_and_initial_points_match_random_search() -> None:
    bounds = [(-5.0, 10.0), (0.0, 15.0)]
    res1 = minimize(branin, bounds, n_calls=12, n_initial=5, seed=42)
    res2 = minimize(branin, bounds, n_calls=12, n_initial=5, seed=42)
    assert isinstance(res1, Result)
    assert res1.xs == res2.xs
    assert res1.ys == res2.ys
    assert res1.x == res2.x
    assert res1.fun == res2.fun

    rs = random_search(branin, bounds, n_calls=12, seed=42)
    assert res1.xs[:5] == rs.xs[:5]
    assert res1.ys[:5] == rs.ys[:5]


def test_on_iteration_callback_contract() -> None:
    history: list[Iteration] = []
    bounds = [(-5.0, 10.0), (0.0, 15.0)]
    res = minimize(
        branin,
        bounds,
        n_calls=10,
        n_initial=4,
        seed=7,
        on_iteration=history.append,
    )
    assert len(history) == 10
    for i, item in enumerate(history):
        assert item.index == i
        assert item.x == res.xs[i]
        assert item.y == res.ys[i]
        if i > 0:
            assert item.best_y <= history[i - 1].best_y
    assert history[-1].best_y == res.fun


def test_error_codes_and_validation() -> None:
    with pytest.raises(OptimizerError) as exc_info:
        minimize(branin, [(1, 1)])
    assert exc_info.value.code == "invalid_bounds"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(branin, [(0.0, 1.0)] * 21)
    assert exc_info.value.code == "invalid_bounds"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(branin, [])
    assert exc_info.value.code == "invalid_bounds"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(branin, [(0.0, float("inf"))])
    assert exc_info.value.code == "invalid_bounds"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(branin, [(1.0,)])  # type: ignore[list-item]
    assert exc_info.value.code == "invalid_bounds"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(branin, [(-5, 10), (0, 15)], n_calls=4, n_initial=5)
    assert exc_info.value.code == "invalid_budget"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(branin, [(-5, 10), (0, 15)], n_calls=501)
    assert exc_info.value.code == "invalid_budget"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(branin, [(-5, 10), (0, 15)], n_calls=10, n_initial=0)
    assert exc_info.value.code == "invalid_budget"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(lambda _: float("nan"), [(0.0, 1.0)], n_calls=3, n_initial=2)
    assert exc_info.value.code == "non_finite_value"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(
            branin,
            [(-5, 10), (0, 15)],
            acquisition="pi",  # type: ignore[arg-type]
        )
    assert exc_info.value.code == "invalid_acquisition"

    with pytest.raises(OptimizerError) as exc_info:
        minimize(branin, [(-5, 10), (0, 15)], xi=float("nan"))
    assert exc_info.value.code == "invalid_acquisition"

    gp = GaussianProcess()
    with pytest.raises(OptimizerError) as exc_info:
        gp.predict(np.array([[0.0, 0.0]]))
    assert exc_info.value.code == "not_fitted"


def test_benchmark_median_beats_random_search_branin_and_hartmann6() -> None:
    branin_bounds = [(-5, 10), (0, 15)]
    rs_branin = [random_search(branin, branin_bounds, n_calls=30, seed=s).fun for s in range(10)]
    ei_branin = [
        minimize(
            branin,
            branin_bounds,
            n_calls=30,
            n_initial=5,
            acquisition="ei",
            seed=s,
        ).fun
        for s in range(10)
    ]
    lcb_branin = [
        minimize(
            branin,
            branin_bounds,
            n_calls=30,
            n_initial=5,
            acquisition="lcb",
            seed=s,
        ).fun
        for s in range(10)
    ]
    assert statistics.median(ei_branin) < statistics.median(rs_branin)
    assert statistics.median(lcb_branin) < statistics.median(rs_branin)

    hartmann_bounds: list[tuple[float, float]] = [(0, 1)] * 6
    rs_hartmann = [
        random_search(hartmann6, hartmann_bounds, n_calls=60, seed=s).fun for s in range(10)
    ]
    ei_hartmann = [
        minimize(
            hartmann6,
            hartmann_bounds,
            n_calls=60,
            n_initial=10,
            acquisition="ei",
            seed=s,
        ).fun
        for s in range(10)
    ]
    lcb_hartmann = [
        minimize(
            hartmann6,
            hartmann_bounds,
            n_calls=60,
            n_initial=10,
            acquisition="lcb",
            seed=s,
        ).fun
        for s in range(10)
    ]
    assert statistics.median(ei_hartmann) < statistics.median(rs_hartmann)
    assert statistics.median(lcb_hartmann) < statistics.median(rs_hartmann)


def test_error_message_truncation() -> None:
    long_msg = "x" * 500
    err = OptimizerError("invalid_bounds", long_msg)
    assert err.code == "invalid_bounds"
    assert len(str(err)) == 200


def test_gp_fit_fails_when_cholesky_always_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _always_fail(*_args: Any, **_kwargs: Any) -> Any:
        raise OptimizerError("numerical_error", "forced failure")

    monkeypatch.setattr("bayesian_optimizer.gp._cholesky_with_jitter", _always_fail)
    gp = GaussianProcess()
    with pytest.raises(OptimizerError) as exc_info:
        gp.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    assert exc_info.value.code == "numerical_error"
