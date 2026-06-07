"""e01-04: Plackett-Burman screening design generation.

Cross-checks pyDOE3's pbdesign against the expected design properties
(2-level, orthogonal, run count divisible by 4).
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import seeds, paths, manifest

import pyDOE3


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e01" / "e01_04_pb"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plackett-Burman for 11 factors -> 12 runs (canonical)
    D = pyDOE3.pbdesign(11)
    runs, factors = D.shape
    levels_ok = bool(set(np.unique(D)) <= {-1.0, 1.0})
    runs_div_4 = (runs % 4 == 0)

    # Orthogonality check: X.T @ X should be diagonal (= n*I) for orthogonal designs
    Xc = D
    gram = Xc.T @ Xc
    diag = np.diag(gram)
    off_diag = gram - np.diag(diag)
    orthogonal = bool(np.allclose(off_diag, 0))
    balanced = bool(np.allclose(diag, runs))

    csv_path = OUT_DIR.parent / "e01_04_pb.csv"
    json_path = OUT_DIR.parent / "e01_04_pb.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["design", "Plackett-Burman 11F"])
        w.writerow(["n_runs", runs])
        w.writerow(["n_factors", factors])
        w.writerow(["runs_divisible_by_4", runs_div_4])
        w.writerow(["levels_are_pm1", levels_ok])
        w.writerow(["balanced (diag == n)", balanced])
        w.writerow(["orthogonal (off-diag == 0)", orthogonal])
        w.writerow(["passes_all", runs_div_4 and levels_ok and balanced and orthogonal])

    json_path.write_text(json.dumps({
        "design": "Plackett-Burman 11F",
        "n_runs": runs, "n_factors": factors,
        "runs_divisible_by_4": runs_div_4,
        "levels_are_pm1": levels_ok,
        "balanced": balanced, "orthogonal": orthogonal,
        "passes_all": bool(runs_div_4 and levels_ok and balanced and orthogonal),
    }, indent=2))

    manifest.write_manifest(
        experiment="e01_04_pb", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"design": "PB-11", "passes": runs_div_4 and levels_ok and orthogonal})

    print(f"  PB shape: {D.shape}   runs%4={runs%4}=0  levels_pm1={levels_ok}")
    print(f"  balanced: {balanced}   orthogonal: {orthogonal}")
    print(f"  PASS: {runs_div_4 and levels_ok and balanced and orthogonal}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
