"""Unit tests for common.models."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import models, designs


def test_linear_ols_recovers_coefficients():
    np.random.seed(0)
    X = np.random.randn(50, 3)
    true_b = np.array([1.0, 2.0, -1.5])
    y = 3.0 + X @ true_b + 0.001 * np.random.randn(50)
    fit = models.fit_linear_ols(X, y)
    assert fit.r2 > 0.99
    assert abs(fit.coefficients[0] - 3.0) < 0.01
    np.testing.assert_allclose(fit.coefficients[1:], true_b, atol=0.01)


def test_scheffe_quadratic_recovers_coefficients():
    SL = designs.simplex_lattice(3, 3)
    true_b = np.array([5.0, 3.0, 7.0, 2.5, -1.5, 0.8])
    Z = np.column_stack([
        SL[:, 0], SL[:, 1], SL[:, 2],
        SL[:, 0] * SL[:, 1], SL[:, 0] * SL[:, 2], SL[:, 1] * SL[:, 2],
    ])
    y = Z @ true_b
    fit = models.fit_scheffe_quadratic(SL, y)
    np.testing.assert_allclose(fit.coefficients, true_b, atol=1e-8)


def test_quadratic_recovers_intercept():
    np.random.seed(1)
    X = np.random.randn(30, 2)
    y = 5.0 + 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 0] ** 2 + 0.001 * np.random.randn(30)
    fit = models.fit_quadratic_ols(X, y)
    assert fit.r2 > 0.99
    assert abs(fit.coefficients[0] - 5.0) < 0.05  # intercept


def test_model_zoo_has_11_families():
    assert len(models.MODEL_ZOO) == 11
    expected = {"linear", "rsm", "scheffe", "sparse_rsm", "pls",
                "bayes_ridge", "gp", "rf", "gbm", "mlp", "ngboost"}
    assert set(models.MODEL_ZOO.keys()) == expected


@pytest.mark.parametrize("name", ["linear", "rsm", "sparse_rsm", "pls",
                                    "bayes_ridge", "gp", "rf", "gbm", "mlp"])
def test_model_zoo_runs_each(name):
    np.random.seed(0)
    X = np.random.randn(40, 3)
    y = 1 + X.sum(axis=1) + X[:, 0] ** 2 + 0.05 * np.random.randn(40)
    fit = models.MODEL_ZOO[name](X, y)
    assert fit.n_obs == 40
    assert fit.r2 > 0.3   # weakest models still reach a sane fit


def test_ensemble_runs():
    np.random.seed(0)
    X = np.random.randn(40, 3)
    y = 1 + X.sum(axis=1) + 0.05 * np.random.randn(40)
    fit = models.fit_ensemble(X, y, seed=0)
    assert fit.r2 > 0.5
    assert len(fit.coefficients) == 4  # 4 ensemble members
