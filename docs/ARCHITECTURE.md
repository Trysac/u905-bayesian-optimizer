# Architecture

## Overview

`bayesian_optimizer` implements deterministic sequential model-based optimization (SMBO) for continuous black-box minimization over box-constrained domains of 1 to 20 dimensions.

## Module Structure

- `bayesian_optimizer.errors`: Defines `OptimizerError` and `ErrorCode` literals.
- `bayesian_optimizer.gp`: Implements `GaussianProcess` with isotropic Matérn 5/2 and RBF kernels, internal `[0, 1]^d` scaling, target standardization, exact log marginal likelihood maximization over `np.geomspace(0.01, 10, 30)`, and Cholesky factorization with jitter escalation.
- `bayesian_optimizer.acquisition`: Implements `expected_improvement` (via vectorized `math.erfc`) and `lower_confidence_bound`.
- `bayesian_optimizer.optimizer`: Implements `minimize`, `random_search`, `Result`, and `Iteration`.
- `bayesian_optimizer.benchmarks`: Implements `branin` and `hartmann6` reference functions.

## Algorithmic Loop

1. Validate bounds (`1 <= dim <= 20`, finite `low < high`) and evaluation budget (`1 <= n_initial <= n_calls <= 500`).
2. Draw `n_initial` uniform random points from `np.random.default_rng(seed)` (matching `random_search` with the same seed).
3. On each subsequent call, fit a `GaussianProcess(kernel="matern52", noise=1e-6)` to all evaluations so far.
4. Draw 2048 uniform candidates from the same RNG, evaluate the acquisition function (`ei` or `lcb`), select the best 5 candidates, and deterministically refine each with 20 coordinate-wise steps of shrinking step size.
5. Evaluate the objective at the highest-scoring refined candidate and invoke `on_iteration` if provided.
