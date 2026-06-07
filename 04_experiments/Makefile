# FORMULA-Sigma paper - experiment harness Makefile
#
# Targets:
#   make all       -> run e01, e02, e03 (full paper rebuild)
#   make e01       -> cross-tool correctness only
#   make e02       -> retrospective reproductions
#   make e03       -> synthetic benchmarks
#   make snapshot  -> recompute expected_hashes.txt from current outputs
#   make check     -> verify outputs against expected_hashes.txt
#   make test      -> pytest unit + smoke tests
#   make clean     -> remove all 05_results/

PY := python

.PHONY: all e01 e02 e03 snapshot check test clean

all: e01 e02 e03

e01:
	$(PY) e01_cross_tool/01_ccd_correctness.py
	$(PY) e01_cross_tool/02_bbd_correctness.py
	$(PY) e01_cross_tool/03_scheffe_mixture.py
	$(PY) e01_cross_tool/04_pbdesign.py
	$(PY) e01_cross_tool/05_lhs_correctness.py
	$(PY) e01_cross_tool/07_full_factorial.py
	$(PY) e01_cross_tool/08_fractional_factorial.py
	$(PY) e01_cross_tool/09_d_optimal.py
	$(PY) e01_cross_tool/10_model_zoo.py
	$(PY) e01_cross_tool/06_summary_table.py

e02:
	$(PY) e02_retrospective/case1_sharma/reproduce.py
	$(PY) e02_retrospective/case1b_akhtar/reproduce.py
	$(PY) e02_retrospective/case2_arif/reproduce.py
	$(PY) e02_retrospective/case3_boscolo/reproduce.py
	$(PY) e02_retrospective/case4_nemr/reproduce.py
	$(PY) e02_retrospective/99_summary_table.py

e03:
	$(PY) e03_synthetic/bo/run_bo_benchmarks.py
	$(PY) e03_synthetic/pds/run_pds_recovery.py
	$(PY) e03_synthetic/robust/run_robust_recovery.py
	$(PY) e03_synthetic/pareto/run_zdt1.py

snapshot:
	$(PY) e04_reproducibility/compute_hashes.py

check:
	$(PY) e04_reproducibility/verify_hashes.py

test:
	$(PY) -m pytest tests/

clean:
	rm -rf "../05_results"
