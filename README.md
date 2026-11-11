# bayesian-optimizer

[![PyPI version](https://img.shields.io/pypi/v/trysac-bayesian-optimizer)](https://pypi.org/project/trysac-bayesian-optimizer/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-bayesian-optimizer)](LICENSE)
[![CI](https://github.com/Trysac/u905-bayesian-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u905-bayesian-optimizer/actions/workflows/ci.yml)

Bayesian optimisation of expensive black-box functions with a hand-written Gaussian process and expected improvement or lower confidence bound

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-bayesian-optimizer
# or
uv add trysac-bayesian-optimizer
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import bayesian_optimizer

# normalize() trims surrounding whitespace and lower-cases the text.
print(bayesian_optimizer.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    bayesian_optimizer.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
