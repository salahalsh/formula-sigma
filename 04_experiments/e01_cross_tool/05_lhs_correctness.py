"""e01-05: Latin hypercube sample sanity vs pyDOE3.

LHS doesn't admit a single canonical numerical target because each draw
is random, but it must satisfy:
  (a) One sample per stratum per dimension
  (b) Uniform marginals on [0,1]
We confirm both for our impl + pyDOE3.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import seeds, designs, paths, manifest

import pyDOE3


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e01" / "e01_05_lhs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N, K = 50, 4
    D_ref = designs.latin_hypercube(N, K, seed=seed)
    D_pydoe = pyDOE3.lhs(K, samples=N, random_state=seed)

    def stratum_test(M, n_bins):
        """Each column has exactly one sample per bin of width 1/n."""
        for j in range(M.shape[1]):
            bins = np.floor(M[:, j] * n_bins).astype(int)
            bins = np.clip(bins, 0, n_bins - 1)
            counts = np.bincount(bins, minlength=n_bins)
            if not np.all(counts == 1):
                return False
        return True

    ok_ours = stratum_test(D_ref, N)
    ok_pydoe = stratum_test(D_pydoe, N)
    range_ours = (float(D_ref.min()), float(D_ref.max()))
    range_pydoe = (float(D_pydoe.min()), float(D_pydoe.max()))

    csv_path = OUT_DIR.parent / "e01_05_lhs.csv"
    json_path = OUT_DIR.parent / "e01_05_lhs.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["design", f"LHS N={N} K={K}"])
        w.writerow(["stratum_correct_ours", ok_ours])
        w.writerow(["stratum_correct_pyDOE3", ok_pydoe])
        w.writerow(["range_ours", str(range_ours)])
        w.writerow(["range_pyDOE3", str(range_pydoe)])
        w.writerow(["passes", ok_ours and ok_pydoe])

    json_path.write_text(json.dumps({
        "design": f"LHS N={N} K={K}",
        "stratum_correct_ours": ok_ours,
        "stratum_correct_pyDOE3": ok_pydoe,
        "range_ours": range_ours,
        "range_pyDOE3": range_pydoe,
        "passes": bool(ok_ours and ok_pydoe),
    }, indent=2))

    manifest.write_manifest(
        experiment="e01_05_lhs", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"design": "LHS", "passes": ok_ours and ok_pydoe})

    print(f"  ours stratum-correct: {ok_ours}   pyDOE3: {ok_pydoe}")
    print(f"  ranges (ours / pyDOE3): {range_ours}  /  {range_pydoe}")
    print(f"  PASS: {ok_ours and ok_pydoe}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
