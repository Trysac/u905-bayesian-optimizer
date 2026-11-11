"""Standard optimization benchmark functions: Branin-Hoo (2D) and Hartmann (6D)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

_HARTMANN6_ALPHA: npt.NDArray[np.float64] = np.array(
    [1.0, 1.2, 3.0, 3.2],
    dtype=np.float64,
)

_HARTMANN6_A: npt.NDArray[np.float64] = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ],
    dtype=np.float64,
)

_HARTMANN6_P: npt.NDArray[np.float64] = 1e-4 * np.array(
    [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ],
    dtype=np.float64,
)


def branin(x1: float | Sequence[float], x2: float | None = None) -> float:
    """Evaluate the 2D Branin-Hoo function.

    Can be called either as ``branin(x1, x2)`` with two floats or ``branin([x1, x2])``
    with a 2-element sequence so it works directly as an objective in ``minimize``.

    Constants:
        a = 1, b = 5.1 / (4 * pi^2), c = 5 / pi, r = 6, s = 10, t = 1 / (8 * pi)
    Global minimum value: ~0.39788735772973816 at (pi, 2.275).
    """
    if x2 is None:
        if isinstance(x1, (int, float)):
            raise TypeError("branin requires either two floats (x1, x2) or a 2-element sequence.")
        if len(x1) != 2:
            raise ValueError("branin sequence input must have length 2.")
        v1 = float(x1[0])
        v2 = float(x1[1])
    else:
        if not isinstance(x1, (int, float)):
            raise TypeError("When x2 is provided, x1 must be a float.")
        v1 = float(x1)
        v2 = float(x2)

    a = 1.0
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    term1 = a * (v2 - b * v1 * v1 + c * v1 - r) ** 2
    term2 = s * (1.0 - t) * math.cos(v1)
    return float(term1 + term2 + s)


def hartmann6(x: Sequence[float]) -> float:
    """Evaluate the 6-dimensional Hartmann function on [0, 1]^6.

    Global minimum value: ~-3.322368011391339 at
    (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573).
    """
    if len(x) != 6:
        raise ValueError("hartmann6 requires a 6-element sequence.")
    x_arr = np.asarray(x, dtype=np.float64)
    diff_sq = np.square(x_arr[np.newaxis, :] - _HARTMANN6_P)
    inner = np.sum(_HARTMANN6_A * diff_sq, axis=1)
    return -float(np.dot(_HARTMANN6_ALPHA, np.exp(-inner)))
