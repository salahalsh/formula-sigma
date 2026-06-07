"""Surrogate-model wrappers.

Fits return a uniform dataclass `FitResult` so e01 / e02 cross-tool
comparisons can be tabulated identically across model families.

The 11 model families in FORMULA-Sigma (one entry per service):
    linear     OLS
    rsm        quadratic / full-RSM polynomial
    scheffe    Scheffe mixture quadratic
    sparse_rsm Lasso on polynomial features
    pls        Partial Least Squares
    bayes_ridge Bayesian Ridge regression
    gp         Gaussian Process
    rf         Random Forest
    gbm        Gradient Boosting
    mlp        Multi-layer Perceptron
    ngboost    Natural-Gradient Boosting (with predictive uncertainty)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import BayesianRidge, Lasso
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass
class FitResult:
    """Container for any surrogate fit."""

    coefficients: np.ndarray
    coefficient_names: Sequence[str]
    r2: float
    rmse: float
    n_obs: int
    n_params: int
    predict_fn: callable
    extra: dict = field(default_factory=dict)


# ---------- Quadratic / RSM via OLS ----------
def fit_quadratic_ols(X: np.ndarray, y: np.ndarray, *, include_interactions: bool = True) -> FitResult:
    """Fit a full quadratic surface y = b0 + sum(bi xi) + sum(bii xi^2) + sum(bij xi xj)."""
    poly = PolynomialFeatures(degree=2, include_bias=False,
                              interaction_only=False)
    Xp = poly.fit_transform(X)
    if not include_interactions:
        # Keep linear + squared only
        names = poly.get_feature_names_out([f"x{i+1}" for i in range(X.shape[1])])
        keep = [i for i, n in enumerate(names) if "^" in n or " " not in n]
        Xp = Xp[:, keep]
        names = names[keep]
    else:
        names = poly.get_feature_names_out([f"x{i+1}" for i in range(X.shape[1])])

    Xp_const = sm.add_constant(Xp)
    model = sm.OLS(y, Xp_const).fit()
    yhat = model.predict(Xp_const)
    coef_names = ["const"] + list(names)
    return FitResult(
        coefficients=model.params,
        coefficient_names=coef_names,
        r2=model.rsquared,
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=len(model.params),
        predict_fn=lambda Xnew: model.predict(sm.add_constant(poly.transform(Xnew))),
        extra={"p_values": model.pvalues, "summary": model.summary().as_text()},
    )


# ---------- Linear-only OLS ----------
def fit_linear_ols(X: np.ndarray, y: np.ndarray) -> FitResult:
    """y = b0 + sum(bi xi)."""
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    yhat = model.predict(Xc)
    names = ["const"] + [f"x{i+1}" for i in range(X.shape[1])]
    return FitResult(
        coefficients=model.params,
        coefficient_names=names,
        r2=model.rsquared,
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=len(model.params),
        predict_fn=lambda Xnew: model.predict(sm.add_constant(Xnew)),
        extra={"p_values": model.pvalues, "summary": model.summary().as_text()},
    )


# ---------- Scheffe mixture (canonical polynomial) ----------
def fit_scheffe_quadratic(X: np.ndarray, y: np.ndarray) -> FitResult:
    """Scheffe quadratic for mixture data.

    y = sum(bi xi) + sum_{i<j} bij xi xj      (no intercept, sum(xi)=1)
    """
    n_comp = X.shape[1]
    cols = [X[:, i] for i in range(n_comp)]
    names = [f"x{i+1}" for i in range(n_comp)]
    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            cols.append(X[:, i] * X[:, j])
            names.append(f"x{i+1}*x{j+1}")
    Z = np.column_stack(cols)
    model = sm.OLS(y, Z).fit()
    yhat = model.predict(Z)
    return FitResult(
        coefficients=model.params,
        coefficient_names=names,
        r2=model.rsquared,
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=len(model.params),
        predict_fn=None,  # caller knows the mixture structure
        extra={"p_values": model.pvalues},
    )


# ---------- Gaussian Process ----------
def fit_gp(X: np.ndarray, y: np.ndarray, *, length_scale: float = 1.0,
           seed: int = 42) -> FitResult:
    """GP surrogate with Matern-5/2 kernel."""
    kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=seed, normalize_y=True,
                                  n_restarts_optimizer=5)
    gp.fit(X, y)
    yhat = gp.predict(X)
    n_params = X.shape[1] + 1  # length scales + amplitude (after optimisation)
    return FitResult(
        coefficients=np.array([gp.kernel_.theta]),
        coefficient_names=["log_kernel_theta"],
        r2=float(1 - np.var(y - yhat) / np.var(y)) if np.var(y) > 0 else 0.0,
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=n_params,
        predict_fn=lambda Xnew, return_std=False: gp.predict(Xnew, return_std=return_std),
        extra={"model": gp, "kernel": gp.kernel_},
    )


# ---------- Sparse RSM (Lasso on polynomial features) ----------
def fit_sparse_rsm(X: np.ndarray, y: np.ndarray, *, alpha: float = 0.01) -> FitResult:
    """Sparse RSM: polynomial features with L1 regularisation."""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X)
    names = list(poly.get_feature_names_out([f"x{i+1}" for i in range(X.shape[1])]))
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(Xp, y)
    yhat = model.predict(Xp)
    return FitResult(
        coefficients=np.concatenate([[model.intercept_], model.coef_]),
        coefficient_names=["const"] + names,
        r2=float(r2_score(y, yhat)),
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=int(np.sum(model.coef_ != 0)) + 1,
        predict_fn=lambda Xnew: model.predict(poly.transform(Xnew)),
        extra={"alpha": alpha, "n_zero_coefs": int(np.sum(model.coef_ == 0))},
    )


# ---------- Partial Least Squares ----------
def fit_pls(X: np.ndarray, y: np.ndarray, *, n_components: int | None = None) -> FitResult:
    """PLS regression for highly collinear designs."""
    n_components = n_components or min(X.shape[0], X.shape[1], 5)
    sc_X = StandardScaler().fit(X)
    Xs = sc_X.transform(X)
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(Xs, y)
    yhat = model.predict(Xs).ravel()
    return FitResult(
        coefficients=model.coef_.ravel(),
        coefficient_names=[f"x{i+1}" for i in range(X.shape[1])],
        r2=float(r2_score(y, yhat)),
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=n_components,
        predict_fn=lambda Xnew: model.predict(sc_X.transform(Xnew)).ravel(),
        extra={"n_components": n_components, "model": model},
    )


# ---------- Bayesian Ridge ----------
def fit_bayes_ridge(X: np.ndarray, y: np.ndarray) -> FitResult:
    """Bayesian Ridge regression on quadratic polynomial features."""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X)
    names = list(poly.get_feature_names_out([f"x{i+1}" for i in range(X.shape[1])]))
    model = BayesianRidge()
    model.fit(Xp, y)
    yhat = model.predict(Xp)
    return FitResult(
        coefficients=np.concatenate([[model.intercept_], model.coef_]),
        coefficient_names=["const"] + names,
        r2=float(r2_score(y, yhat)),
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=len(model.coef_) + 1,
        predict_fn=lambda Xnew, return_std=False:
            model.predict(poly.transform(Xnew), return_std=return_std),
        extra={"alpha_posterior": float(model.alpha_),
               "lambda_posterior": float(model.lambda_)},
    )


# ---------- Random Forest ----------
def fit_rf(X: np.ndarray, y: np.ndarray, *, n_estimators: int = 200, seed: int = 42) -> FitResult:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                  n_jobs=1)
    model.fit(X, y)
    yhat = model.predict(X)
    return FitResult(
        coefficients=model.feature_importances_,
        coefficient_names=[f"x{i+1}_importance" for i in range(X.shape[1])],
        r2=float(r2_score(y, yhat)),
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=n_estimators,
        predict_fn=lambda Xnew: model.predict(Xnew),
        extra={"n_estimators": n_estimators, "model": model},
    )


# ---------- Gradient Boosting Machine ----------
def fit_gbm(X: np.ndarray, y: np.ndarray, *,
            n_estimators: int = 200, learning_rate: float = 0.05,
            max_depth: int = 3, seed: int = 42) -> FitResult:
    model = GradientBoostingRegressor(n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth,
                                      random_state=seed)
    model.fit(X, y)
    yhat = model.predict(X)
    return FitResult(
        coefficients=model.feature_importances_,
        coefficient_names=[f"x{i+1}_importance" for i in range(X.shape[1])],
        r2=float(r2_score(y, yhat)),
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=n_estimators,
        predict_fn=lambda Xnew: model.predict(Xnew),
        extra={"n_estimators": n_estimators, "lr": learning_rate, "max_depth": max_depth},
    )


# ---------- Multi-Layer Perceptron ----------
def fit_mlp(X: np.ndarray, y: np.ndarray, *,
            hidden_layer_sizes=(32, 16), max_iter: int = 2000,
            seed: int = 42) -> FitResult:
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                         max_iter=max_iter, random_state=seed,
                         early_stopping=False)
    model.fit(Xs, y)
    yhat = model.predict(Xs)
    n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    return FitResult(
        coefficients=np.array([np.nan]),  # not interpretable
        coefficient_names=["mlp_weights_n_params"],
        r2=float(r2_score(y, yhat)),
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=int(n_params),
        predict_fn=lambda Xnew: model.predict(sc.transform(Xnew)),
        extra={"hidden_layers": list(hidden_layer_sizes), "n_params": int(n_params)},
    )


# ---------- NGBoost (with uncertainty) ----------
def fit_ngboost(X: np.ndarray, y: np.ndarray, *,
                n_estimators: int = 200, learning_rate: float = 0.05,
                seed: int = 42) -> FitResult:
    from ngboost import NGBRegressor
    model = NGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                         random_state=seed, verbose=False)
    model.fit(X, y)
    yhat = model.predict(X)
    return FitResult(
        coefficients=np.array([np.nan]),
        coefficient_names=["ngboost_internal"],
        r2=float(r2_score(y, yhat)),
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=n_estimators,
        predict_fn=lambda Xnew, return_std=False: (
            (model.predict(Xnew), model.pred_dist(Xnew).std())
            if return_std else model.predict(Xnew)
        ),
        extra={"n_estimators": n_estimators, "lr": learning_rate, "uncertainty": True},
    )


# ---------- Ensemble (weighted average of model zoo predictions) ----------
def fit_ensemble(X: np.ndarray, y: np.ndarray, *, seed: int = 42) -> FitResult:
    """Weighted average of (quadratic OLS, RF, GBM, GP).

    Weights are 1/RMSE on in-sample fit (toy heuristic; production
    FORMULA-Sigma uses out-of-bag or held-out RMSE).
    """
    fits = {
        "rsm":  fit_quadratic_ols(X, y),
        "rf":   fit_rf(X, y, seed=seed),
        "gbm":  fit_gbm(X, y, seed=seed),
        "gp":   fit_gp(X, y, seed=seed),
    }
    weights = np.array([1.0 / max(f.rmse, 1e-9) for f in fits.values()])
    weights /= weights.sum()
    preds = np.column_stack([f.predict_fn(X) for f in fits.values()])
    yhat = preds @ weights
    return FitResult(
        coefficients=weights,
        coefficient_names=list(fits.keys()),
        r2=float(r2_score(y, yhat)),
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
        n_obs=len(y),
        n_params=sum(f.n_params for f in fits.values()),
        predict_fn=lambda Xnew: np.column_stack(
            [f.predict_fn(Xnew) for f in fits.values()]) @ weights,
        extra={"weights": dict(zip(fits.keys(), weights.tolist())),
               "members": list(fits.keys())},
    )


# ---------- Registry of all 11 model families ----------
MODEL_ZOO = {
    "linear":     fit_linear_ols,
    "rsm":        fit_quadratic_ols,
    "scheffe":    fit_scheffe_quadratic,
    "sparse_rsm": fit_sparse_rsm,
    "pls":        fit_pls,
    "bayes_ridge": fit_bayes_ridge,
    "gp":         fit_gp,
    "rf":         fit_rf,
    "gbm":        fit_gbm,
    "mlp":        fit_mlp,
    "ngboost":    fit_ngboost,
    # ensemble is meta-model; tested separately
}
