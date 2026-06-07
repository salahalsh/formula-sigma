"""End-to-end smoke tests: each experiment script runs without crashing."""
import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(ROOT / script)],
                            capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, f"{script} failed:\n{result.stderr[-1500:]}"


def test_e01_ccd():
    _run("e01_cross_tool/01_ccd_correctness.py")


def test_e01_bbd():
    _run("e01_cross_tool/02_bbd_correctness.py")


def test_e01_scheffe():
    _run("e01_cross_tool/03_scheffe_mixture.py")


def test_e02_case1():
    _run("e02_retrospective/case1_sharma/reproduce.py")


def test_e02_case2():
    _run("e02_retrospective/case2_arif/reproduce.py")


def test_e01_full_factorial():
    _run("e01_cross_tool/07_full_factorial.py")


def test_e01_fractional():
    _run("e01_cross_tool/08_fractional_factorial.py")


def test_e01_d_optimal():
    _run("e01_cross_tool/09_d_optimal.py")


def test_e01_model_zoo():
    _run("e01_cross_tool/10_model_zoo.py")
