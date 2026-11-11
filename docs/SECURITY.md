# Security

## Threat Model

### Defended

- **Input Validation**: Enforces dimensional limits (1-20), valid budget (n_calls <= 500), and finite bounds.
- **Error Handling**: `OptimizerError` prevents information leaks by truncating error messages to 200 characters.
- **Numerical Stability**: Cholesky factorization utilizes iterative jitter escalation (`1e-10` to `1e-4`) to recover from singular matrices.

### Not Defended

- **Objective Function Safety**: The library executes the user-provided objective function in the same process; the user is responsible for sandboxing arbitrary execution.
- **Denial of Service**: The optimization loop is bounded, but an extremely slow objective function can still cause process hang, as the library does not implement per-call timeouts.