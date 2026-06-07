"""e03 Pareto: NSGA-II on ZDT1 - hypervolume vs known Pareto front."""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import seeds, paths, manifest, optimisers


def zdt1(X):
    n = X.shape[1]
    f1 = X[:, 0]
    g = 1 + 9.0 * np.sum(X[:, 1:], axis=1) / (n - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.column_stack([f1, f2])


def hypervolume(F, ref):
    """2-D hypervolume vs a reference point (assumes minimisation)."""
    F = F[np.lexsort((F[:, 1], F[:, 0]))]
    hv = 0.0
    prev_x = 0.0
    prev_y = ref[1]
    for x, y in F:
        if x >= ref[0] or y >= ref[1]:
            continue
        hv += (x - prev_x) * (prev_y - y)
        prev_x = x; prev_y = y
    hv += (ref[0] - prev_x) * (prev_y - ref[1])
    # Above formulation is approximate; simpler 2D HV via Klee 's measure follows
    # Compute exactly by integrating along sorted-by-x and tracking min-y so-far:
    F = F[np.argsort(F[:, 0])]
    hv = 0.0
    y_prev = ref[1]
    for x, y in F:
        if x >= ref[0] or y >= ref[1]:
            continue
        hv += max(0, y_prev - y) * (ref[0] - x)
        y_prev = min(y_prev, y)
    return hv


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e03" / "pareto"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_VAR = 6
    bounds = np.column_stack([np.zeros(N_VAR), np.ones(N_VAR)])
    REF = np.array([1.1, 1.1])

    res = optimisers.nsga2_pareto(
        objective_fn=zdt1, n_var=N_VAR, bounds=bounds, n_obj=2,
        n_gen=80, pop_size=80, seed=seed,
    )
    F = res.F

    # True Pareto front for ZDT1: f2 = 1 - sqrt(f1), f1 in [0, 1]
    f1_true = np.linspace(0, 1, 200)
    f2_true = 1.0 - np.sqrt(f1_true)
    true_front = np.column_stack([f1_true, f2_true])

    hv_obs = hypervolume(F, REF)
    hv_true = hypervolume(true_front, REF)
    ratio = hv_obs / hv_true if hv_true > 0 else float("nan")

    csv_path = OUT_DIR / "pareto_summary.csv"
    json_path = OUT_DIR / "pareto_detail.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["test_problem", "ZDT1 (n=6)"])
        w.writerow(["n_pareto_points", len(F)])
        w.writerow(["hypervolume_observed", round(hv_obs, 5)])
        w.writerow(["hypervolume_true_front", round(hv_true, 5)])
        w.writerow(["hypervolume_ratio", round(ratio, 5)])
        w.writerow(["passes_at_0.95", ratio > 0.95])

    json_path.write_text(json.dumps({
        "test_problem": "ZDT1",
        "n_var": N_VAR,
        "n_pareto_points": int(len(F)),
        "hypervolume_observed": float(hv_obs),
        "hypervolume_true": float(hv_true),
        "hypervolume_ratio": float(ratio),
        "passes_at_0.95": bool(ratio > 0.95),
    }, indent=2))

    manifest.write_manifest(
        experiment="e03_pareto_zdt1", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0)

    print(f"  ZDT1 n={N_VAR}: {len(F)} Pareto points")
    print(f"  HV observed:   {hv_obs:.5f}")
    print(f"  HV true front: {hv_true:.5f}")
    print(f"  HV ratio:      {ratio:.5f}    PASS at 0.95: {ratio > 0.95}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
