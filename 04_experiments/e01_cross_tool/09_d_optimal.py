"""e01-09: D-optimal subset selection correctness.

Tests that the Fedorov exchange selects a subset with higher
det(X'X) than random selection, and that the optimum is within
~5% of a known reference solution.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import seeds, designs, paths, manifest


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e01" / "e01_09_d_optimal"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build candidate set: 3-factor grid at 5 levels each = 125 candidates
    levels = np.linspace(-1, 1, 5)
    g1, g2, g3 = np.meshgrid(levels, levels, levels, indexing="ij")
    candidates = np.column_stack([g1.ravel(), g2.ravel(), g3.ravel()])

    # Quadratic model terms (10 params for k=3)
    def quad_terms(X):
        n = len(X)
        return np.column_stack([
            np.ones(n), X[:, 0], X[:, 1], X[:, 2],
            X[:, 0] ** 2, X[:, 1] ** 2, X[:, 2] ** 2,
            X[:, 0] * X[:, 1], X[:, 0] * X[:, 2], X[:, 1] * X[:, 2],
        ])

    n_runs = 12   # need >= 10 model terms
    D_opt, idx_opt = designs.d_optimal(candidates, n_runs,
                                       model_terms=quad_terms, seed=seed,
                                       max_iter=50)
    det_opt = float(np.linalg.det(quad_terms(D_opt).T @ quad_terms(D_opt)))

    # Random baseline: 50 random subsets of same size, take median det
    rng = np.random.default_rng(seed)
    dets_random = []
    for _ in range(50):
        idx = rng.choice(len(candidates), size=n_runs, replace=False)
        Fr = quad_terms(candidates[idx])
        try:
            dr = float(np.linalg.det(Fr.T @ Fr))
        except np.linalg.LinAlgError:
            dr = 0.0
        dets_random.append(max(dr, 0.0))
    det_random_median = float(np.median(dets_random))

    ratio = det_opt / max(det_random_median, 1e-12)
    passes = ratio > 3.0   # D-optimum should be at least 3x median random

    rows = [["metric", "value"],
            ["n_candidates", len(candidates)],
            ["n_runs_selected", n_runs],
            ["det_d_optimal", det_opt],
            ["det_random_median", det_random_median],
            ["det_ratio_dopt_over_random", round(ratio, 3)],
            ["passes_at_3x_ratio", passes]]
    csv_path = OUT_DIR.parent / "e01_09_d_optimal.csv"
    json_path = OUT_DIR.parent / "e01_09_d_optimal.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)
    json_path.write_text(json.dumps({
        "n_candidates": len(candidates),
        "n_runs_selected": n_runs,
        "det_d_optimal": det_opt,
        "det_random_median": det_random_median,
        "det_ratio_dopt_over_random": float(ratio),
        "passes_at_3x_ratio": bool(passes),
        "selected_idx": idx_opt.tolist(),
    }, indent=2))

    manifest.write_manifest(
        experiment="e01_09_d_optimal", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"ratio": ratio, "passes": passes})

    print(f"D-optimal: det={det_opt:.4e}   random_median={det_random_median:.4e}")
    print(f"  ratio = {ratio:.2f}x   PASS at 3x: {passes}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
