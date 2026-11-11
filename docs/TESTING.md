# Testing

## Structure

Tests are organized to mirror the source module structure, ensuring 100% path coverage for logic:

- `tests/test_bayesian_optimizer.py`: Covers the full API, benchmarks (`branin`, `hartmann6`), and validation errors.

## Coverage

- The test suite covers acceptance criteria including: exact matching of benchmark minima, deterministic behavior across seeds, and proper handling of all `OptimizerError` codes.
- Determinism is verified by comparing results of `minimize` calls using the same seed.
- Input validation ensures `bounds` and `budget` constraints are strictly checked.