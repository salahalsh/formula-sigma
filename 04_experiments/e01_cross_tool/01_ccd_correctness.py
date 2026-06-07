"""e01-01: CCD design generation + OLS fit agreement.

Compares three sources for a 3-factor rotatable CCD:
  - common.designs.ccd_rotatable  (our reference, numpy only)
  - pyDOE3.ccdesign                (third-party)
  - statsmodels OLS on a synthetic quadratic response

Output:
  05_results/e01/e01_01_ccd.csv    (correctness table for Table 2 row 1)
  05_results/e01/e01_01_ccd.json   (full numerical results)
  05_results/e01/e01_01_ccd/run_manifest.json
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import seeds, designs, paths, manifest

import pyDOE3


def _name_features(n_factors, X_fit):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X_fit)
    names = poly.get_feature_names_out([f"x{i+1}" for i in range(n_factors)])
    return poly, names


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e01" / "e01_01_ccd"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_FACTORS = 3
    N_CENTER  = 6
    EXPECTED_ALPHA = 2.0 ** (N_FACTORS / 4.0)   # 1.6818 for k=3

    # ---- Design generation ----
    D_ref = designs.ccd_rotatable(N_FACTORS, n_center=N_CENTER)
    D_pydoe = pyDOE3.ccdesign(N_FACTORS, center=(0, N_CENTER), alpha="r", face="ccc")

    alpha_ref = float(np.max(np.abs(D_ref)))
    alpha_pydoe = float(np.max(np.abs(D_pydoe)))

    # Sort both for set-equality check
    def canon(M):
        return np.array(sorted(map(tuple, np.round(M, 8))))

    set_equal = np.array_equal(canon(D_ref), canon(D_pydoe))

    # ---- Synthetic response and fit ----
    rng = np.random.default_rng(seed)
    true_coef = {"b0": 5.0, "b1": 2.0, "b2": -1.5, "b3": 0.8,
                 "b11": 1.2, "b22": -0.4, "b33": 0.6,
                 "b12": 0.5, "b13": -0.3, "b23": 0.2}
    sigma = 0.05

    def truth(X):
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return (true_coef["b0"] + true_coef["b1"] * x1 + true_coef["b2"] * x2
                + true_coef["b3"] * x3 + true_coef["b11"] * x1**2
                + true_coef["b22"] * x2**2 + true_coef["b33"] * x3**2
                + true_coef["b12"] * x1*x2 + true_coef["b13"] * x1*x3
                + true_coef["b23"] * x2*x3)

    y = truth(D_ref) + rng.normal(0, sigma, size=len(D_ref))

    # Fit with statsmodels on poly features
    poly, names = _name_features(N_FACTORS, D_ref)
    Xp = sm.add_constant(poly.transform(D_ref))
    fit_sm = sm.OLS(y, Xp).fit()

    # Also fit with numpy lstsq (independent reference)
    coef_np, residuals_np, rank, _ = np.linalg.lstsq(Xp, y, rcond=None)

    # Coefficient agreement
    max_abs_diff_sm_vs_np = float(np.max(np.abs(fit_sm.params - coef_np)))
    max_rel_diff_sm_vs_np = float(np.max(np.abs(fit_sm.params - coef_np) /
                                          (np.abs(fit_sm.params) + 1e-12)))

    # ---- Save outputs ----
    csv_path = OUT_DIR.parent / "e01_01_ccd.csv"
    json_path = OUT_DIR.parent / "e01_01_ccd.json"

    import csv as _csv
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["design", "CCD rotatable 3F"])
        w.writerow(["n_runs_total", len(D_ref)])
        w.writerow(["n_center", N_CENTER])
        w.writerow(["alpha_expected", round(EXPECTED_ALPHA, 6)])
        w.writerow(["alpha_observed_ref", round(alpha_ref, 6)])
        w.writerow(["alpha_observed_pyDOE3", round(alpha_pydoe, 6)])
        w.writerow(["design_matrices_equal_setwise", set_equal])
        w.writerow(["fit_R2", round(fit_sm.rsquared, 8)])
        w.writerow(["fit_RMSE", round(float(np.sqrt(np.mean(fit_sm.resid**2))), 6)])
        w.writerow(["max_abs_coef_diff_sm_vs_numpy", max_abs_diff_sm_vs_np])
        w.writerow(["max_rel_coef_diff_sm_vs_numpy", max_rel_diff_sm_vs_np])
        w.writerow(["agreement_threshold_1e-6", max_rel_diff_sm_vs_np < 1e-6])

    payload = {
        "design": "CCD rotatable 3F",
        "n_runs": len(D_ref),
        "n_center": N_CENTER,
        "alpha_expected": EXPECTED_ALPHA,
        "alpha_observed_ref": alpha_ref,
        "alpha_observed_pyDOE3": alpha_pydoe,
        "design_matrices_equal_setwise": set_equal,
        "true_coefficients": true_coef,
        "fitted_coefficients_statsmodels": dict(zip(["const"] + list(names),
                                                    fit_sm.params.tolist())),
        "fitted_coefficients_numpy_lstsq": dict(zip(["const"] + list(names),
                                                    coef_np.tolist())),
        "fit_R2": float(fit_sm.rsquared),
        "fit_RMSE": float(np.sqrt(np.mean(fit_sm.resid ** 2))),
        "max_abs_coef_diff_sm_vs_numpy": max_abs_diff_sm_vs_np,
        "max_rel_coef_diff_sm_vs_numpy": max_rel_diff_sm_vs_np,
        "passes_1e-6": max_rel_diff_sm_vs_np < 1e-6,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    manifest.write_manifest(
        experiment="e01_01_ccd",
        out_dir=OUT_DIR,
        seed=seed,
        inputs=[],
        outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"design": "CCD rotatable 3F", "passes": bool(payload["passes_1e-6"])},
    )

    print(f"  alpha (expected vs ref vs pyDOE3): "
          f"{EXPECTED_ALPHA:.6f} | {alpha_ref:.6f} | {alpha_pydoe:.6f}")
    print(f"  set-equal: {set_equal}")
    print(f"  fit R^2 = {fit_sm.rsquared:.8f}   RMSE = {np.sqrt(np.mean(fit_sm.resid**2)):.6f}")
    print(f"  max |sm - numpy| coefficient diff (relative): {max_rel_diff_sm_vs_np:.2e}")
    print(f"  PASS at 1e-6: {payload['passes_1e-6']}")
    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")


if __name__ == "__main__":
    main()
