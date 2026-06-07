# 03_data/processed - per-case extracted CSVs

Machine-readable extractions from the source PMC XMLs. Every case
folder follows a consistent structure:

```
case<N>_<slug>/
  table_NN.csv               raw table dumps (audit only)
  case<N>_design.csv         clean factor matrix
  case<N>_responses.csv      per-run responses + SD
  case<N>_design_and_responses.csv     merged convenience
  case<N>_known_optimum.csv  published optimum + verification
  case<N>_ingestion.yaml     factors, responses, anomalies, scope
  [case-specific extras]     coefficients, kinetics, ANOVA, stability
```

## Live cases

| Folder                          | Design family                        | Runs |
|---------------------------------|--------------------------------------|------|
| case1_tablet_ccd/               | rotatable CCD                        | 20   |
| case1b_akhtar_bilayer/          | bilayer dual rotatable CCD           | 10+10|
| case2_nanoparticle_mixture/     | D-optimal combined mixture-process   | 22   |
| case3_lyophilization_bbd/       | Box-Behnken                          | 15   |
| case4_optimization/             | D-optimal w/ categorical             | 22   |

All design + response CSVs are deterministic outputs of:
- `D:\tmp\fx_build_case1_sharma.py`     (Sharma)
- `D:\tmp\fx_build_case1b_akhtar.py`    (Akhtar 1b)
- `D:\tmp\fx_build_case_packs.py`       (Arif, Boscolo, Nemr)

## _audit/ - obsolete extractions kept for audit

- `_audit/obsolete_picks/case1_kotamarthy/` - Kotamarthy CSVs built
  before swap. Includes a `case1_responses.csv` with ranges + named
  extreme runs that documents WHY full reproduction was infeasible.
