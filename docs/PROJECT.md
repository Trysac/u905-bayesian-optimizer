# Project record

## Objective

Bayesian optimisation of expensive black-box functions with a hand-written Gaussian process and expected improvement or lower confidence bound.

## Scope

- The library provides `minimize` (BO), `random_search` (baseline), `GaussianProcess` (GP), and standard benchmark functions `branin` and `hartmann6`.
- Typed errors and deterministic iteration logging.

## Out of Scope

- Categorical or integer parameters.
- Parallel or batch suggestions.
- Multi-objective optimization.
- Constraints.
- An "ask-tell" server.

## Effort metrics

| Metric | Value |
|---|---|
| Tokens (input / output) | 349177 / 15998 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.4836 |

## Factory run u905 (measured from API usage)

| Date (UTC) | Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | generation | google/gemini-4-argon | yes | 5635 | 14170 | 0 | 0 | 0.3059 |
| 2026-11-11 | audit | google/gemini-3.8-flash | no | 181975 | 690 | 0 | 20070 | 0.1406 |
| 2026-11-11 | docs | google/gemini-3.1-flash-lite | no | 141497 | 1138 | 0 | 0 | 0.0371 |

Total: $0.4836. Prices: vertex line, as measured by its adapter. Providers serve the current version of each model id, so the date tells which version answered; versions, launch stages and price sources per line are in the factory's model register (docs/MODELS.md).
