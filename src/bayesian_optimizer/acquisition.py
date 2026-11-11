"""Acquisition functions for Bayesian optimization (minimization)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

_ERFC_VEC = np.vectorize(math.erfc, otypes=[np.float64])
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Standard normal cumulative distribution function."""
    res: npt.NDArray[np.float64] = np.array(0.5 * _ERFC_VEC(-z * _INV_SQRT2), dtype=np.float64)
    return res


def _norm_pdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Standard normal probability density function."""
    res: npt.NDArray[np.float64] = np.array(
        _INV_SQRT2PI * np.exp(-0.5 * np.square(z)), dtype=np.float64
    )
    return res


def expected_improvement(
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
    best: float,
    xi: float = 0.01,
) -> npt.NDArray[np.float64]:
    """Compute Expected Improvement (EI) for minimization.

    Formula:
        z = (best - mu - xi) / sigma
        EI = (best - mu - xi) * Phi(z) + sigma * phi(z) where sigma > 0, else 0.
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    ei = np.zeros_like(mu_arr, dtype=np.float64)
    mask = sigma_arr > 0.0
    if np.any(mask):
        imp = best - mu_arr[mask] - xi
        s = sigma_arr[mask]
        z = imp / s
        ei[mask] = imp * _norm_cdf(z) + s * _norm_pdf(z)
        ei[mask] = np.maximum(0.0, ei[mask])
    return ei


def lower_confidence_bound(
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
    kappa: float = 2.0,
) -> npt.NDArray[np.float64]:
    """Compute Lower Confidence Bound (LCB) for minimization: mu - kappa * sigma."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    return mu_arr - kappa * sigma_arr
