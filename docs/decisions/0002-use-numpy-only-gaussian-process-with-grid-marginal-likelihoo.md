# Use NumPy-only Gaussian Process with Grid Marginal Likelihood and Coordinate Refinement

Status: accepted (generated, pending human review)

## Context

Full-featured Bayesian optimization libraries depend on SciPy, scikit-learn, or PyTorch and use quasi-Newton solvers that can vary across platforms or BLAS implementations. We need a small, auditable, strictly deterministic optimizer that beats random search on standard benchmarks.

## Decision

Depend only on numpy (>=2,<3). Implement the normal CDF via math.erfc vectorized with np.vectorize, select the GP kernel length-scale by maximizing the exact log marginal likelihood over a fixed 30-point geometric grid (np.geomspace(0.01, 10, 30)) with Cholesky jitter escalation (1e-10 to 1e-4), and optimize the acquisition function by sampling 2048 uniform candidates followed by 20 deterministic coordinate-wise refinement steps on the top 5 candidates.

## Consequences

Zero SciPy or scikit-learn dependency footprint, bitwise-reproducible runs for a given seed on a platform, straightforward code auditing, and strong sample efficiency on low-to-medium dimensional black-box functions up to 20 dimensions and 500 evaluations.
