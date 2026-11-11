"""Gaussian Process regressor with Matérn 5/2 and RBF kernels."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from bayesian_optimizer.errors import OptimizerError

KernelName = Literal["matern52", "rbf"]

_LENGTH_SCALE_GRID: npt.NDArray[np.float64] = np.geomspace(0.01, 10.0, 30, dtype=np.float64)
_SQRT5 = math.sqrt(5.0)
_MAX_SAMPLES = 2000
_MAX_DIMS = 20


def _pairwise_distances(
    xa: npt.NDArray[np.float64],
    xb: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute Euclidean pairwise distance matrix between rows of xa and xb."""
    diff = xa[:, np.newaxis, :] - xb[np.newaxis, :, :]
    sq_dist: npt.NDArray[np.float64] = np.asarray(
        np.sum(np.square(diff), axis=-1), dtype=np.float64
    )
    return np.asarray(np.sqrt(np.maximum(0.0, sq_dist)), dtype=np.float64)


def _compute_kernel(
    dist: npt.NDArray[np.float64],
    length_scale: float,
    kernel: KernelName,
) -> npt.NDArray[np.float64]:
    """Evaluate isotropic Matérn 5/2 or RBF kernel with signal variance 1.0."""
    if kernel == "matern52":
        scaled = (_SQRT5 * dist) / length_scale
        poly = 1.0 + scaled + (5.0 * np.square(dist)) / (3.0 * length_scale * length_scale)
        return np.asarray(poly * np.exp(-scaled), dtype=np.float64)
    scaled_sq = np.square(dist / length_scale)
    return np.asarray(np.exp(-0.5 * scaled_sq), dtype=np.float64)


def _cholesky_with_jitter(
    k_mat: npt.NDArray[np.float64],
    noise: float,
) -> npt.NDArray[np.float64]:
    """Compute lower-triangular Cholesky factor with jitter escalation from 1e-10 to 1e-4."""
    n = k_mat.shape[0]
    base = k_mat.copy()
    diag_idx = np.diag_indices(n)
    jitter = 1e-10
    while jitter <= 1e-4 + 1e-15:
        mat = base.copy()
        mat[diag_idx] += noise + jitter
        try:
            chol: npt.NDArray[np.float64] = np.asarray(np.linalg.cholesky(mat), dtype=np.float64)
            return chol
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise OptimizerError(
        "numerical_error",
        "Cholesky decomposition failed after escalating jitter up to 1e-4.",
    )


def _solve_cholesky(
    chol: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve (chol @ chol.T) @ x = rhs via numpy linear solver on triangular factors."""
    z = np.linalg.solve(chol, rhs)
    res: npt.NDArray[np.float64] = np.asarray(np.linalg.solve(chol.T, z), dtype=np.float64)
    return res


class GaussianProcess:
    """Exact Gaussian Process regressor with unit-cube input scaling and target standardization."""

    def __init__(
        self,
        *,
        kernel: KernelName = "matern52",
        noise: float = 1e-6,
    ) -> None:
        if kernel not in ("matern52", "rbf"):
            raise OptimizerError(
                "numerical_error",
                f"Unsupported kernel: {kernel!r}. Expected 'matern52' or 'rbf'.",
            )
        if not math.isfinite(noise) or noise < 0.0:
            raise OptimizerError(
                "numerical_error",
                "Noise variance must be a non-negative finite float.",
            )
        self.kernel: KernelName = kernel
        self.noise: float = float(noise)
        self._fitted: bool = False
        self._x_min: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._x_range: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._x_scaled: npt.NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._length_scale: float = 1.0
        self._chol: npt.NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)
        self._alpha: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        bounds: npt.ArrayLike | None = None,
    ) -> GaussianProcess:
        """Fit the Gaussian Process model to observations (X, y)."""
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if x_arr.ndim != 2 or y_arr.ndim != 1:
            raise OptimizerError(
                "numerical_error",
                "X must be 2D and y must be 1D.",
            )
        n_samples, n_dims = x_arr.shape
        if n_samples == 0 or n_samples != y_arr.shape[0] or n_samples > _MAX_SAMPLES:
            raise OptimizerError(
                "numerical_error",
                "Invalid number of training samples for GaussianProcess.",
            )
        if n_dims < 1 or n_dims > _MAX_DIMS:
            raise OptimizerError(
                "invalid_bounds",
                f"Dimensions must be between 1 and {_MAX_DIMS}.",
            )
        if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
            raise OptimizerError(
                "non_finite_value",
                "Training inputs X and targets y must contain only finite values.",
            )

        if bounds is not None:
            b_arr = np.asarray(bounds, dtype=np.float64)
            x_min = b_arr[:, 0]
            x_max = b_arr[:, 1]
        else:
            x_min = np.min(x_arr, axis=0)
            x_max = np.max(x_arr, axis=0)

        x_range = x_max - x_min
        x_range = np.where(x_range > 0.0, x_range, 1.0)
        x_scaled = (x_arr - x_min) / x_range

        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr))
        if y_std <= 1e-12:
            y_std = 1.0
        y_scaled = (y_arr - y_mean) / y_std

        dist = _pairwise_distances(x_scaled, x_scaled)

        best_lml = -math.inf
        best_ls = float(_LENGTH_SCALE_GRID[0])
        best_chol: npt.NDArray[np.float64] | None = None
        best_alpha: npt.NDArray[np.float64] | None = None

        for ls_val in _LENGTH_SCALE_GRID:
            ls = float(ls_val)
            k_mat = _compute_kernel(dist, ls, self.kernel)
            try:
                chol = _cholesky_with_jitter(k_mat, self.noise)
            except OptimizerError:
                continue
            alpha = _solve_cholesky(chol, y_scaled)
            log_det = float(np.sum(np.log(np.diag(chol))))
            quad = float(np.dot(y_scaled, alpha))
            lml = -0.5 * quad - log_det - 0.5 * n_samples * math.log(2.0 * math.pi)
            if lml > best_lml:
                best_lml = lml
                best_ls = ls
                best_chol = chol
                best_alpha = alpha

        if best_chol is None or best_alpha is None:
            raise OptimizerError(
                "numerical_error",
                "Failed to fit GaussianProcess across all candidate length scales.",
            )

        self._x_min = x_min
        self._x_range = x_range
        self._x_scaled = x_scaled
        self._y_mean = y_mean
        self._y_std = y_std
        self._length_scale = best_ls
        self._chol = best_chol
        self._alpha = best_alpha
        self._fitted = True
        return self

    def predict(
        self,
        X: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Predict posterior mean and standard deviation for query points X."""
        if not self._fitted:
            raise OptimizerError(
                "not_fitted",
                "GaussianProcess must be fitted with fit(X, y) before calling predict(X).",
            )
        x_arr = np.asarray(X, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[1] != self._x_scaled.shape[1]:
            raise OptimizerError(
                "numerical_error",
                "Query array X must be 2D with matching feature dimension.",
            )
        if not np.all(np.isfinite(x_arr)):
            raise OptimizerError(
                "non_finite_value",
                "Query inputs X must contain only finite values.",
            )

        x_scaled = (x_arr - self._x_min) / self._x_range
        dist = _pairwise_distances(x_scaled, self._x_scaled)
        k_star = _compute_kernel(dist, self._length_scale, self.kernel)

        mean_scaled = k_star @ self._alpha
        mean = self._y_mean + self._y_std * mean_scaled

        v = np.linalg.solve(self._chol, k_star.T)
        var_scaled = np.maximum(0.0, 1.0 - np.sum(np.square(v), axis=0))
        std = self._y_std * np.sqrt(var_scaled)
        return mean, std
