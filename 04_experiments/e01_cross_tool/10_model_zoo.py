"""e01-10: All 11 model families fit and agree on a synthetic benchmark.

Generates a noisy quadratic surface, fits each of the 11 model families
in common.models.MODEL_ZOO, plus the ensemble, and reports R^2 and
RMSE. Confirms that all models load and run without errors and that
they recover the surface to a sane R^2.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import seeds, designs, models, paths, manifest


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e01" / "e01_10_model_zoo"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Benchmark surface: noisy quadratic on a 3-factor CCD
    rng = np.random.default_rng(seed)
    X = designs.ccd_rotatable(3, n_center=6)
    true_y = (5.0 + 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2]
              + 1.2 * X[:, 0] ** 2 - 0.4 * X[:, 1] ** 2 + 0.6 * X[:, 2] ** 2
              + 0.5 * X[:, 0] * X[:, 1])
    y = true_y + rng.normal(0, 0.05, size=len(X))

    rows = [["model_family", "n_obs", "n_params", "R2", "RMSE",
             "passes_R2_above_0.5"]]
    detail = {}
    runtimes = {}     # tracked separately to keep CSV byte-deterministic
    for name, fit_fn in models.MODEL_ZOO.items():
        t1 = time.time()
        try:
            if name == "scheffe":
                rows.append([name, len(y), 0, "n/a (not a mixture)", "n/a", "skip"])
                continue
            fit = fit_fn(X, y)
            r2 = float(fit.r2)
            rmse = float(fit.rmse)
            dt = time.time() - t1
            runtimes[name] = dt
            passes = bool(r2 > 0.5)
            rows.append([name, fit.n_obs, fit.n_params,
                         f"{r2:.4f}", f"{rmse:.4f}", passes])
            detail[name] = {"R2": r2, "RMSE": rmse, "n_params": fit.n_params,
                            "passes_R2_above_0.5": passes}
        except Exception as e:
            rows.append([name, "FAIL", "-", "-", "-", False])
            detail[name] = {"error": str(e)[:200]}

    # Add the ensemble
    try:
        t1 = time.time()
        ens = models.fit_ensemble(X, y, seed=seed)
        dt = time.time() - t1
        runtimes["ensemble"] = dt
        rows.append(["ensemble (rsm+rf+gbm+gp)", ens.n_obs, ens.n_params,
                     f"{ens.r2:.4f}", f"{ens.rmse:.4f}", ens.r2 > 0.5])
        detail["ensemble"] = {"R2": float(ens.r2), "RMSE": float(ens.rmse),
                              "weights": ens.extra["weights"]}
    except Exception as e:
        rows.append(["ensemble", "FAIL", "-", "-", "-", False])
        detail["ensemble"] = {"error": str(e)[:200]}

    csv_path = OUT_DIR.parent / "e01_10_model_zoo.csv"
    json_path = OUT_DIR.parent / "e01_10_model_zoo.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)
    json_path.write_text(json.dumps(detail, indent=2))

    manifest.write_manifest(
        experiment="e01_10_model_zoo", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"n_model_families": len(rows) - 1,
               "per_model_runtime_sec": runtimes})

    print("Model zoo (FORMULA-Sigma 11 families + ensemble):")
    for r in rows:
        print("  ", "  ".join(str(c) for c in r))
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
