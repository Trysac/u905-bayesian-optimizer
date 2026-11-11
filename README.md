# bayesian-optimizer

Bayesian optimisation of expensive black-box functions with a hand-written Gaussian process and expected improvement or lower confidence bound.

## Installation

```bash
uv add bayesian-optimizer
```

## Usage

```python
from bayesian_optimizer import Iteration, minimize
from bayesian_optimizer.benchmarks import branin


def log_step(it: Iteration) -> None:
    print(f"step={it.index:02d} y={it.y:.4f} best={it.best_y:.4f}")


res = minimize(branin, [(-5.0, 10.0), (0.0, 15.0)], n_calls=15, seed=0, on_iteration=log_step)
print(f"Best: {res.fun:.6f} at {res.x}")
```

## Public API Summary

- `minimize(...)`: Bayesian optimization loop.
- `random_search(...)`: Uniform random search baseline.
- `GaussianProcess(...)`: Exact GP regressor with Mat'ern 5/2 or RBF kernels.
- `expected_improvement(...)` / `lower_confidence_bound(...)`: Acquisition functions.
- `Result`: Dataclass containing optimization history and best finding.
- `Iteration`: Progress snapshot.
- `OptimizerError`: Typed exception with codes `invalid_bounds`, `invalid_budget`, `non_finite_value`, `invalid_acquisition`, `not_fitted`, `numerical_error`.

Limits: 1-20 dimensions, 1 <= n_initial <= n_calls <= 500.

## Why not <best-known alternative>

Existing Bayesian optimisation packages (like scikit-optimize or Optuna) pull in large scientific stacks and are difficult to audit. This library provides a single small dependency, readable GP implementation, and fully deterministic runs useful for teaching and auditing.

See [docs/SECURITY.md](docs/SECURITY.md) for details on threat modeling.