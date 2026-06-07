"""e03 BO: convergence on Branin, Hartmann-6, Ackley.

Compares BO (GP + Expected Improvement) vs random search vs LHS+single-GP.
10 seeds per condition; reports median simple regret over budget.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np
from skopt import gp_minimize, dummy_minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import seeds, paths, manifest


# ---------- Test functions ----------
def branin(x):
    """Branin (2D), global min ~0.398 at (-pi, 12.275), (pi, 2.275), (9.425, 2.475)."""
    x1, x2 = float(x[0]), float(x[1])
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

BRANIN_BOUNDS = [(-5.0, 10.0), (0.0, 15.0)]
BRANIN_MIN = 0.397887


def hartmann6(x):
    """Hartmann-6 (6D), global min -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    x = np.asarray(x, dtype=float)
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i]) ** 2)
        outer += alpha[i] * np.exp(-inner)
    return float(-outer)

HARTMANN6_BOUNDS = [(0.0, 1.0)] * 6
HARTMANN6_MIN = -3.32237


def ackley(x):
    """Ackley (4D)."""
    x = np.asarray(x, dtype=float)
    a, b, c = 20.0, 0.2, 2 * np.pi
    n = len(x)
    s1 = np.sum(x ** 2) / n
    s2 = np.sum(np.cos(c * x)) / n
    return float(-a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + np.e)

ACKLEY_BOUNDS = [(-2.0, 2.0)] * 4
ACKLEY_MIN = 0.0


def run_method(method, fn, bounds, n_calls, seed):
    if method == "BO":
        res = gp_minimize(fn, bounds, n_calls=n_calls, n_initial_points=10,
                          random_state=seed, acq_func="EI", verbose=False)
    elif method == "RAND":
        res = dummy_minimize(fn, bounds, n_calls=n_calls, random_state=seed,
                             verbose=False)
    else:
        raise ValueError(method)
    return np.asarray(res.func_vals)


def simple_regret_history(values, true_min):
    running_best = np.minimum.accumulate(values)
    return running_best - true_min


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e03" / "bo"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bench = [
        ("Branin",     branin,     BRANIN_BOUNDS,     BRANIN_MIN,     30),
        ("Hartmann-6", hartmann6,  HARTMANN6_BOUNDS,  HARTMANN6_MIN,  80),
        ("Ackley-4",   ackley,     ACKLEY_BOUNDS,     ACKLEY_MIN,     60),
    ]
    N_SEEDS = 5  # keep modest for runtime; bump in production
    summary = [["test_function", "n_dim", "budget", "method",
                "median_simple_regret_at_budget",
                "iqr_simple_regret_at_budget"]]
    detail = {}

    for name, fn, bounds, true_min, budget in bench:
        detail[name] = {"budget": budget, "true_min": true_min, "methods": {}}
        for method in ["BO", "RAND"]:
            regrets = []
            for s in range(N_SEEDS):
                vals = run_method(method, fn, bounds, n_calls=budget, seed=seed + s)
                regrets.append(simple_regret_history(vals, true_min))
            R = np.array(regrets)
            final = R[:, -1]
            summary.append([
                name, len(bounds), budget, method,
                f"{float(np.median(final)):.4f}",
                f"{float(np.percentile(final, 75) - np.percentile(final, 25)):.4f}",
            ])
            detail[name]["methods"][method] = {
                "n_seeds": N_SEEDS,
                "median_regret_history": np.median(R, axis=0).tolist(),
                "p25_regret_history":    np.percentile(R, 25, axis=0).tolist(),
                "p75_regret_history":    np.percentile(R, 75, axis=0).tolist(),
                "final_median": float(np.median(final)),
                "final_iqr": float(np.percentile(final, 75) - np.percentile(final, 25)),
            }
            print(f"  {name:10s} {method:5s}: median_final_regret = {np.median(final):.4f}")

    csv_path = OUT_DIR / "bo_summary.csv"
    json_path = OUT_DIR / "bo_detail.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(summary)
    json_path.write_text(json.dumps(detail, indent=2))

    manifest.write_manifest(
        experiment="e03_bo_benchmarks", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"n_seeds": N_SEEDS, "test_functions": [b[0] for b in bench]})

    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")


if __name__ == "__main__":
    main()
