# 04_experiments - FORMULA-Sigma Validation Harness

The four experiments that populate Section 3 of the manuscript.

## Quick start

```
make all       # rebuild e01, e02, e03
make test      # run pytest unit + smoke
make snapshot  # snapshot expected_hashes.txt
make check     # verify outputs against expected_hashes.txt
```

Direct invocation works too:

```
python e01_cross_tool/01_ccd_correctness.py
```

## Layout

```
04_experiments/
├── pyproject.toml          # dependency list
├── Makefile                # one-line entry points
├── README.md               # this file
├── common/                 # shared library (designs, models, optimisers, PDS, IO)
├── e01_cross_tool/         # 5 scripts: CCD, BBD, Scheffe, PB, LHS + summary
├── e02_retrospective/      # 5 case studies + summary
│   ├── case1_sharma/
│   ├── case1b_akhtar/
│   ├── case2_arif/
│   ├── case3_boscolo/
│   └── case4_nemr/
├── e03_synthetic/          # BO, PDS, robust, Pareto
│   ├── bo/
│   ├── pds/
│   ├── robust/
│   └── pareto/
├── e04_reproducibility/    # hash snapshot + verify + make_paper.sh
└── tests/                  # pytest unit + smoke
```

## Outputs

All deterministic outputs land under `../05_results/`:

```
05_results/
├── e01/        per-experiment CSV + JSON + run_manifest.json
├── e02/        per-case summary + detail + diagnostic + manifest
├── e03/        per-method summary + JSON + manifest
├── tables_raw/ Table2 (cross-tool) + Table3 (retrospective summary)
└── figures_raw/ raw figure exports before final manuscript styling
```

## Determinism

- Every experiment calls `common.seeds.set_seeds(42)` before running.
- Every experiment writes `run_manifest.json` capturing python version,
  library versions, seed, input/output SHA-256, runtime.
- `e04_reproducibility/expected_hashes.txt` snapshots SHA-256 of all
  deterministic outputs (excluding manifests which carry timestamps).
- `python e04_reproducibility/verify_hashes.py` checks current outputs
  against the snapshot.

## Adding FORMULA-Sigma as a third comparison column

The `common/` package is structured so a thin FORMULA-Sigma adapter can
be added without touching individual experiment scripts. Implementation
hook lives at `common/models.py` near `fit_quadratic_ols` - copy that
function shape and call into FORMULA-Sigma's REST API or Python client.

## Reference tools used as baselines

| Library        | Role                                                            |
|----------------|-----------------------------------------------------------------|
| pyDOE3 2.x     | Cross-check design generation (CCD, BBD, PB, LHS)               |
| statsmodels    | Canonical OLS reference for coefficient comparison              |
| scikit-learn   | GP, RF, NN surrogates; PolynomialFeatures                       |
| scikit-optimize| Bayesian optimisation baseline (GP-EI)                          |
| pymoo          | NSGA-II Pareto baseline                                         |
| scipy          | differential evolution for robust optimum search                |
