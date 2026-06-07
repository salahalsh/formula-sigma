"""Optimisation wrappers: desirability, NSGA-II Pareto, robust, Bayesian."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# ---------- Derringer-Suich desirability ----------
@dataclass
class DesirabilityResult:
    x: np.ndarray
    desirability: float
    per_response: dict


def desirability_one_sided(y: np.ndarray, low: float, high: float,
                           weight: float = 1.0, goal: str = "max") -> np.ndarray:
    """Compute desirability d_i in [0, 1].

    goal: 'max' (low<=L=>d=0, low>=H=>d=1, linear in between)
          'min' (mirror)
          'target' uses target=midpoint(low,high)
    """
    y = np.asarray(y, dtype=float)
    d = np.zeros_like(y)
    if goal == "max":
        in_range = (y >= low) & (y <= high)
        d[in_range] = ((y[in_range] - low) / (high - low)) ** weight
        d[y > high] = 1.0
    elif goal == "min":
        in_range = (y >= low) & (y <= high)
        d[in_range] = ((high - y[in_range]) / (high - low)) ** weight
        d[y < low] = 1.0
    elif goal == "target":
        mid = 0.5 * (low + high)
        in_range_lo = (y >= low) & (y <= mid)
        in_range_hi = (y > mid) & (y <= high)
        d[in_range_lo] = ((y[in_range_lo] - low) / (mid - low)) ** weight
        d[in_range_hi] = ((high - y[in_range_hi]) / (high - mid)) ** weight
    return d


def overall_desirability(desirabilities: Sequence[np.ndarray]) -> np.ndarray:
    """Geometric mean across responses."""
    arr = np.column_stack(desirabilities)
    return np.exp(np.mean(np.log(np.clip(arr, 1e-12, 1.0)), axis=1))


# ---------- True NSGA-II multi-objective via pymoo ----------
def nsga2_pareto(objective_fn: Callable[[np.ndarray], np.ndarray],
                 n_var: int,
                 bounds: np.ndarray,
                 n_obj: int,
                 n_gen: int = 100,
                 pop_size: int = 50,
                 seed: int = 42):
    """Run NSGA-II on a vectorised objective."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=bounds[:, 0],
                             xu=bounds[:, 1])

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = objective_fn(X)

    res = minimize(_P(), NSGA2(pop_size=pop_size), ("n_gen", n_gen),
                   seed=seed, verbose=False)
    return res


# ---------- Robust optimum search ----------
def robust_optimum(predict_fn: Callable[[np.ndarray], np.ndarray],
                   n_var: int,
                   bounds: np.ndarray,
                   noise_sigma: np.ndarray,
                   n_mc: int = 200,
                   goal: str = "max",
                   seed: int = 42) -> dict:
    """Find x* that maximises E[y(x + noise)] over input noise."""
    from scipy.optimize import differential_evolution

    rng = np.random.default_rng(seed)

    def neg_robust(x):
        noise = rng.normal(0.0, noise_sigma, size=(n_mc, len(x)))
        Xs = x[None, :] + noise
        ys = predict_fn(Xs)
        return -float(np.mean(ys)) if goal == "max" else float(np.mean(ys))

    result = differential_evolution(neg_robust, list(map(tuple, bounds)),
                                    seed=seed, polish=True, tol=1e-6,
                                    maxiter=100)
    return {
        "x_robust": result.x,
        "robust_expected_y": -result.fun if goal == "max" else result.fun,
        "scipy_result": result,
    }


# ---------- Bayesian optimisation ----------
def bayes_opt(objective_fn: Callable[[np.ndarray], float],
              bounds: list[tuple[float, float]],
              n_calls: int = 30,
              n_initial: int = 10,
              seed: int = 42) -> dict:
    """skopt-based BO with GP-EI acquisition.

    objective_fn must accept a single point (1-D array) and return a scalar
    to MINIMISE. Convert max problems by negating.
    """
    from skopt import gp_minimize

    res = gp_minimize(
        func=lambda v: float(objective_fn(np.asarray(v))),
        dimensions=list(bounds),
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=seed,
        acq_func="EI",
    )
    return {
        "x_best": np.asarray(res.x),
        "f_best": float(res.fun),
        "history_x": np.asarray(res.x_iters),
        "history_f": np.asarray(res.func_vals),
    }
