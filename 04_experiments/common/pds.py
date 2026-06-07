"""Probabilistic design space (M5) - Monte Carlo over a fitted surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class PDSResult:
    """Probabilistic design space result.

    grid     -> evaluated factor grid (n_points, n_factors)
    prob     -> P(all spec satisfied) per grid point
    spec     -> the (low, high) spec window per response
    n_mc     -> Monte Carlo replicates used
    """
    grid: np.ndarray
    prob: np.ndarray
    spec: list[tuple[float, float]]
    n_mc: int


def monte_carlo_pds(predict_fn: Callable[[np.ndarray], np.ndarray],
                    noise_fn: Callable[[int, int], np.ndarray],
                    spec: Sequence[tuple[float, float]],
                    grid: np.ndarray,
                    n_mc: int = 1000,
                    seed: int = 42) -> PDSResult:
    """Compute P(spec satisfied) at each grid point via Monte Carlo.

    predict_fn(X) -> (n, n_resp)   deterministic surface predictions
    noise_fn(n, n_resp) -> (n, n_resp)   residual noise sampler
    spec[k] = (low_k, high_k)               spec window per response
    grid                                    factor grid to evaluate

    Returns probability of meeting ALL spec windows simultaneously per grid pt.
    """
    rng = np.random.default_rng(seed)
    mu = predict_fn(grid)                       # (n_grid, n_resp)
    if mu.ndim == 1:
        mu = mu[:, None]
    n_grid, n_resp = mu.shape
    successes = np.zeros(n_grid, dtype=int)
    for _ in range(n_mc):
        noise = noise_fn(n_grid, n_resp)
        y = mu + noise
        ok = np.ones(n_grid, dtype=bool)
        for k, (lo, hi) in enumerate(spec):
            ok &= (y[:, k] >= lo) & (y[:, k] <= hi)
        successes += ok.astype(int)
    return PDSResult(
        grid=grid, prob=successes / n_mc, spec=list(spec), n_mc=n_mc,
    )


def grid_2d(bounds: np.ndarray, n: int = 50) -> np.ndarray:
    """2D meshgrid suitable for visualisation."""
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], n)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], n)
    X1, X2 = np.meshgrid(x1, x2)
    return np.column_stack([X1.ravel(), X2.ravel()])
