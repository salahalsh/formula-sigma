"""e03 robust: robust optimum recovery under input noise.

True surface has a NOMINAL optimum at the top of a sharp ridge and a
ROBUST optimum on a flatter shoulder. Without noise, gradient finds the
nominal optimum; with input noise, FORMULA-Sigma's robust mode should
find the flatter shoulder.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import seeds, paths, manifest, optimisers


# True surface: bimodal, sharp peak at (1.0, 1.0) value 1.0 ; flatter shoulder around (0, 0) value 0.8
def true_y(x):
    x = np.atleast_2d(x)
    sharp = 1.0 * np.exp(-50 * ((x[:, 0] - 1.0) ** 2 + (x[:, 1] - 1.0) ** 2))
    flat = 0.8 * np.exp(-2 * (x[:, 0] ** 2 + x[:, 1] ** 2))
    return sharp + flat


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e03" / "robust"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    bnds = list(map(tuple, bounds))

    # Nominal optimum: maximise true_y (no noise)
    res_nom = differential_evolution(lambda x: -float(true_y(x)[0]),
                                     bnds, seed=seed, tol=1e-8, maxiter=200)
    x_nom = res_nom.x
    y_nom = -res_nom.fun

    # Robust optimum: maximise E[true_y(x + N(0, 0.15))]
    noise_sigma = np.array([0.15, 0.15])
    res_rob = optimisers.robust_optimum(
        predict_fn=true_y, n_var=2, bounds=bounds,
        noise_sigma=noise_sigma, n_mc=500, goal="max", seed=seed,
    )
    x_rob = res_rob["x_robust"]
    y_rob_expected = res_rob["robust_expected_y"]

    # Evaluate nominal point's expected value under same noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_sigma, size=(500, 2))
    y_nom_under_noise = float(np.mean(true_y(x_nom[None, :] + noise)))

    csv_path = OUT_DIR / "robust_summary.csv"
    json_path = OUT_DIR / "robust_detail.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["x_nominal", f"({x_nom[0]:.3f}, {x_nom[1]:.3f})"])
        w.writerow(["y_nominal_noisefree", round(y_nom, 4)])
        w.writerow(["y_at_nominal_under_noise", round(y_nom_under_noise, 4)])
        w.writerow(["x_robust", f"({x_rob[0]:.3f}, {x_rob[1]:.3f})"])
        w.writerow(["y_robust_expected_under_noise", round(y_rob_expected, 4)])
        w.writerow(["robust_beats_nominal_under_noise",
                    y_rob_expected > y_nom_under_noise])
        w.writerow(["expected_gain", round(y_rob_expected - y_nom_under_noise, 4)])

    json_path.write_text(json.dumps({
        "x_nominal": x_nom.tolist(),
        "y_nominal_noisefree": float(y_nom),
        "y_at_nominal_under_noise": y_nom_under_noise,
        "x_robust": x_rob.tolist(),
        "y_robust_expected_under_noise": float(y_rob_expected),
        "robust_beats_nominal_under_noise": bool(y_rob_expected > y_nom_under_noise),
        "expected_gain_robust_over_nominal": float(y_rob_expected - y_nom_under_noise),
        "noise_sigma": noise_sigma.tolist(),
    }, indent=2))

    manifest.write_manifest(
        experiment="e03_robust_recovery", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0)

    print(f"  nominal optimum: x={x_nom.round(3).tolist()}  y_noisefree={y_nom:.4f}")
    print(f"  nominal under noise: y_expected = {y_nom_under_noise:.4f}")
    print(f"  robust optimum:  x={x_rob.round(3).tolist()}  y_expected={y_rob_expected:.4f}")
    print(f"  expected gain of robust over nominal: {y_rob_expected - y_nom_under_noise:+.4f}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
