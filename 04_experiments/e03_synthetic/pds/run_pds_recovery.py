"""e03 PDS: probabilistic design space recovery on a known RSM surface.

Builds a 2D quadratic surface with analytical spec-failure boundary,
generates noisy observations, fits FORMULA-Sigma-style PDS via Monte
Carlo, and reports Hausdorff distance between analytical and recovered
0.95-probability boundary.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import seeds, paths, manifest, pds, plotting


# Known surface y(x1,x2) = b0 + b1 x1 + b2 x2 + b11 x1^2 + b22 x2^2 + b12 x1 x2
TRUE = {"b0": 5.0, "b1": 2.0, "b2": -1.5, "b11": -1.2, "b22": -0.8, "b12": 0.3}
SIGMA = 0.5
SPEC = (3.0, 7.0)


def truth(X):
    return (TRUE["b0"] + TRUE["b1"] * X[:, 0] + TRUE["b2"] * X[:, 1]
            + TRUE["b11"] * X[:, 0] ** 2 + TRUE["b22"] * X[:, 1] ** 2
            + TRUE["b12"] * X[:, 0] * X[:, 1])


def hausdorff(pts_a, pts_b):
    """Symmetric Hausdorff distance between two point sets."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("nan")
    d_a = np.min(np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1), axis=1).max()
    d_b = np.min(np.linalg.norm(pts_b[:, None, :] - pts_a[None, :, :], axis=-1), axis=1).max()
    return float(max(d_a, d_b))


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e03" / "pds"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate data on a 13-run CCD
    rng = np.random.default_rng(seed)
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    from common import designs
    Xd = designs.ccd_rotatable(2, n_center=5)
    yd = truth(Xd) + rng.normal(0, SIGMA, size=len(Xd))

    # Fit quadratic
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(Xd)
    fit = sm.OLS(yd, sm.add_constant(Xp)).fit()
    res_sigma = float(np.sqrt(fit.mse_resid))

    # Predict grid
    grid = pds.grid_2d(bounds, n=60)
    grid_p = poly.transform(grid)
    grid_c = sm.add_constant(grid_p, has_constant="add")

    def predict_fn(X):
        Xq = sm.add_constant(poly.transform(X), has_constant="add")
        return fit.predict(Xq)[:, None]

    def noise_fn(n, k):
        return rng.normal(0.0, res_sigma, size=(n, k))

    result = pds.monte_carlo_pds(predict_fn, noise_fn, [SPEC], grid,
                                 n_mc=500, seed=seed)

    # Analytical boundary: P(y in spec) = 0.95
    # y_true ~ N(mu(x), SIGMA^2), so P(spec_low <= y <= spec_high) is a
    # function of mu(x). Solve for mu range giving prob = 0.95.
    from scipy.stats import norm
    # Find mu range where prob >= 0.95
    mus = np.linspace(0.0, 10.0, 2000)
    probs_analytical = norm.cdf((SPEC[1] - mus) / SIGMA) - norm.cdf((SPEC[0] - mus) / SIGMA)
    mu_lo = mus[np.argmax(probs_analytical >= 0.95)]
    mu_hi = mus[len(probs_analytical) - 1 - np.argmax(probs_analytical[::-1] >= 0.95)]
    if mu_lo >= mu_hi:
        mu_lo = mu_hi = 0.5 * (SPEC[0] + SPEC[1])

    # Analytical boundary on grid: where TRUE(grid) is in [mu_lo, mu_hi]
    mu_grid = truth(grid)
    analytic_in = (mu_grid >= mu_lo) & (mu_grid <= mu_hi)
    recovered_in = result.prob >= 0.95

    # Hausdorff between boundaries (in grid units, normalized)
    def boundary(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.empty((0, 2))
        return grid[idx]

    H = hausdorff(boundary(analytic_in), boundary(recovered_in))
    # Jaccard similarity is a more interpretable region-overlap metric
    inter = int(np.sum(analytic_in & recovered_in))
    union = int(np.sum(analytic_in | recovered_in))
    jaccard = inter / union if union else 0.0

    # ---- Plot ----
    plotting.apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    cmap = "viridis"
    ext = [bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1]]
    P_grid = result.prob.reshape(60, 60)
    axes[0].imshow(P_grid, extent=ext, origin="lower", cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axes[0].contour(np.linspace(*bounds[0], 60), np.linspace(*bounds[1], 60), P_grid,
                    levels=[0.95], colors=plotting.WONG["red"], linewidths=2)
    axes[0].set_title("FORMULA-Sigma PDS  (P>=0.95 in red)")
    axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")

    A_grid = analytic_in.reshape(60, 60).astype(float)
    axes[1].imshow(A_grid, extent=ext, origin="lower", cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axes[1].contour(np.linspace(*bounds[0], 60), np.linspace(*bounds[1], 60), A_grid,
                    levels=[0.5], colors=plotting.WONG["red"], linewidths=2)
    axes[1].set_title(f"Analytical PDS  (Hausdorff = {H:.3f})")
    axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2")
    plotting.save_both(fig, "e03_pds_recovery", paths.FIGURES_RAW)

    # ---- Save outputs ----
    csv_path = OUT_DIR / "pds_summary.csv"
    json_path = OUT_DIR / "pds_detail.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["n_design_pts", len(Xd)])
        w.writerow(["fit_R2", round(fit.rsquared, 4)])
        w.writerow(["residual_sigma", round(res_sigma, 4)])
        w.writerow(["spec_low", SPEC[0]])
        w.writerow(["spec_high", SPEC[1]])
        w.writerow(["analytical_mu_low_for_0.95_prob", round(mu_lo, 4)])
        w.writerow(["analytical_mu_high_for_0.95_prob", round(mu_hi, 4)])
        w.writerow(["analytic_region_size", int(analytic_in.sum())])
        w.writerow(["recovered_region_size", int(recovered_in.sum())])
        w.writerow(["hausdorff_distance_grid_units", round(H, 4)])
        w.writerow(["intersection_cells", inter])
        w.writerow(["union_cells", union])
        w.writerow(["jaccard_similarity", round(jaccard, 4)])
        w.writerow(["passes_at_0.80_jaccard", jaccard > 0.80])

    json_path.write_text(json.dumps({
        "spec": list(SPEC),
        "fit_R2": float(fit.rsquared),
        "residual_sigma": res_sigma,
        "analytic_mu_window": [mu_lo, mu_hi],
        "n_analytic": int(analytic_in.sum()),
        "n_recovered": int(recovered_in.sum()),
        "hausdorff_grid_units": H,
        "intersection_cells": inter,
        "union_cells": union,
        "jaccard_similarity": jaccard,
        "passes_at_0.80_jaccard": bool(jaccard > 0.80),
    }, indent=2))

    manifest.write_manifest(
        experiment="e03_pds_recovery", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path,
                            paths.FIGURES_RAW / "e03_pds_recovery.pdf"],
        runtime_sec=time.time() - t0,
        extra={"hausdorff": H})

    print(f"  fit R^2 = {fit.rsquared:.4f}   residual sigma = {res_sigma:.4f}")
    print(f"  analytic region |#cells: {analytic_in.sum()}/{len(grid)}")
    print(f"  recovered region #cells: {recovered_in.sum()}/{len(grid)}")
    print(f"  Hausdorff distance: {H:.4f}   intersection/union = {inter}/{union}")
    print(f"  Jaccard similarity: {jaccard:.4f}   PASS at 0.80: {jaccard > 0.80}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
