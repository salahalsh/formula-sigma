"""e01-03: Scheffe simplex-lattice mixture design + Scheffe quadratic fit.

Cross-checks our simplex-lattice {q=3, m=3} design against textbook
counts and our Scheffe quadratic fit against statsmodels.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import math
import numpy as np
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import seeds, designs, models, paths, manifest


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e01" / "e01_03_scheffe"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Simplex-lattice {3, 3} should have C(3+3-1, 3-1) = C(5,2) = 10 points
    SL = designs.simplex_lattice(3, 3)
    expected_count = math.comb(3 + 3 - 1, 3 - 1)
    sum_one = bool(np.allclose(SL.sum(axis=1), 1.0))

    # Synthetic Scheffe quadratic: y = b1*x1 + b2*x2 + b3*x3 + b12*x1*x2 + b13*x1*x3 + b23*x2*x3
    rng = np.random.default_rng(seed)
    true = np.array([5.0, 3.0, 7.0, 2.5, -1.5, 0.8])
    Z = np.column_stack([
        SL[:, 0], SL[:, 1], SL[:, 2],
        SL[:, 0]*SL[:, 1], SL[:, 0]*SL[:, 2], SL[:, 1]*SL[:, 2],
    ])
    y = Z @ true + rng.normal(0, 0.01, size=len(SL))

    # Our wrapper
    fit_ours = models.fit_scheffe_quadratic(SL, y)

    # Reference: numpy lstsq through the same Z (no intercept)
    coef_np, *_ = np.linalg.lstsq(Z, y, rcond=None)

    max_rel = float(np.max(np.abs(fit_ours.coefficients - coef_np) /
                           (np.abs(fit_ours.coefficients) + 1e-12)))

    csv_path = OUT_DIR.parent / "e01_03_scheffe.csv"
    json_path = OUT_DIR.parent / "e01_03_scheffe.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["design", "Scheffe simplex-lattice {3,3}"])
        w.writerow(["n_runs_observed", len(SL)])
        w.writerow(["n_runs_expected_combinatorial", expected_count])
        w.writerow(["row_sums_eq_one", sum_one])
        w.writerow(["fit_R2", round(fit_ours.r2, 8)])
        w.writerow(["fit_RMSE", round(fit_ours.rmse, 6)])
        w.writerow(["max_rel_coef_diff_ours_vs_numpy", max_rel])
        w.writerow(["passes_1e-6", max_rel < 1e-6])

    json_path.write_text(json.dumps({
        "design": "Scheffe simplex-lattice {3,3}",
        "n_runs_observed": len(SL),
        "n_runs_expected_combinatorial": expected_count,
        "row_sums_eq_one": sum_one,
        "true_coefficients": dict(zip(["b1","b2","b3","b12","b13","b23"], true.tolist())),
        "fitted_coefficients_ours": dict(zip(fit_ours.coefficient_names,
                                              fit_ours.coefficients.tolist())),
        "fitted_coefficients_numpy_lstsq": coef_np.tolist(),
        "fit_R2": float(fit_ours.r2),
        "fit_RMSE": float(fit_ours.rmse),
        "max_rel_coef_diff_ours_vs_numpy": max_rel,
        "passes_1e-6": bool(max_rel < 1e-6),
    }, indent=2))

    manifest.write_manifest(
        experiment="e01_03_scheffe", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"design": "Scheffe {3,3}", "passes": bool(max_rel < 1e-6)})

    print(f"  Scheffe lattice {{3,3}} count: observed={len(SL)} expected={expected_count}")
    print(f"  row sums equal to 1: {sum_one}")
    print(f"  fit R^2 = {fit_ours.r2:.8f}")
    print(f"  max rel coef diff (ours vs numpy): {max_rel:.2e}")
    print(f"  PASS at 1e-6: {max_rel < 1e-6}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
