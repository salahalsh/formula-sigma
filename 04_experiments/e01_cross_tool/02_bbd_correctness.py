"""e01-02: BBD design generation + OLS fit agreement.

Compares our reference 3-factor BBD against pyDOE3's bbdesign.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import seeds, designs, paths, manifest

import pyDOE3


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e01" / "e01_02_bbd"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    D_ref = designs.bbd_3factor(n_center=3)
    D_pydoe = pyDOE3.bbdesign(3, center=3)

    def canon(M):
        return np.array(sorted(map(tuple, np.round(M, 8))))

    set_equal = np.array_equal(canon(D_ref), canon(D_pydoe))

    # Synthetic quadratic
    rng = np.random.default_rng(seed)
    truth = lambda X: 10 + 2*X[:,0] - X[:,1] + 0.5*X[:,2] \
                          + X[:,0]**2 - 0.3*X[:,1]**2 + 0.2*X[:,0]*X[:,1]
    y = truth(D_ref) + rng.normal(0, 0.03, size=len(D_ref))

    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = sm.add_constant(poly.fit_transform(D_ref))
    fit = sm.OLS(y, Xp).fit()
    coef_np, *_ = np.linalg.lstsq(Xp, y, rcond=None)

    max_rel = float(np.max(np.abs(fit.params - coef_np) /
                           (np.abs(fit.params) + 1e-12)))

    csv_path = OUT_DIR.parent / "e01_02_bbd.csv"
    json_path = OUT_DIR.parent / "e01_02_bbd.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["design", "Box-Behnken 3F"])
        w.writerow(["n_runs", len(D_ref)])
        w.writerow(["design_matrices_equal_setwise", set_equal])
        w.writerow(["fit_R2", round(fit.rsquared, 8)])
        w.writerow(["fit_RMSE", round(float(np.sqrt(np.mean(fit.resid**2))), 6)])
        w.writerow(["max_rel_coef_diff_sm_vs_numpy", max_rel])
        w.writerow(["passes_1e-6", max_rel < 1e-6])

    json_path.write_text(json.dumps({
        "design": "Box-Behnken 3F",
        "n_runs": len(D_ref),
        "design_matrices_equal_setwise": set_equal,
        "fitted_coefficients_statsmodels": fit.params.tolist(),
        "fitted_coefficients_numpy_lstsq": coef_np.tolist(),
        "fit_R2": float(fit.rsquared),
        "fit_RMSE": float(np.sqrt(np.mean(fit.resid**2))),
        "max_rel_coef_diff_sm_vs_numpy": max_rel,
        "passes_1e-6": bool(max_rel < 1e-6),
    }, indent=2))

    manifest.write_manifest(
        experiment="e01_02_bbd", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"design": "BBD 3F", "passes": bool(max_rel < 1e-6)})

    print(f"  set-equal vs pyDOE3: {set_equal}")
    print(f"  fit R^2 = {fit.rsquared:.8f}")
    print(f"  max rel coef diff sm vs numpy: {max_rel:.2e}")
    print(f"  PASS at 1e-6: {max_rel < 1e-6}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
